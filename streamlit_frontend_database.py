import streamlit as st
from langgraph_backend_database import llm, chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid # For generating unique thread IDs

#*****************************************************Utility Functions*****************************************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
    if state.values and 'messages' in state.values:
        return state.values['messages']
    return []  # Return empty list if no messages found

#*****************************************************Session Setup*****************************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

#*****************************************************Sidebar UI*****************************************************
st.sidebar.title("Chatbot")
if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:  # Show latest threads on top
    if st.sidebar.button(str(thread_id), key=f"thread_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()  # Force rerender with the new conversation

#*****************************************************Main Chat UI*****************************************************

# Display conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    with st.chat_message('user'):
        st.markdown(user_input)

    # Get conversation history for context
    messages = []
    for msg in st.session_state['message_history']:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))

    # Stream response directly from LLM
    with st.chat_message('assistant'):
        response_placeholder = st.empty()
        full_response = ""
        
        # Direct streaming from LLM - this WILL stream word by word
        for chunk in llm.stream(messages):
            if chunk.content:
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    # Save to history
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
    
    # Save both user message and AI response to LangGraph for thread persistence
    CONFIG = {'configurable': {'thread_id': str(st.session_state['thread_id'])}}
    chatbot.invoke({'messages': [HumanMessage(content=user_input), AIMessage(content=full_response)]}, config=CONFIG)
