import streamlit as st
from langgraph_backend import llm, chatbot
from langchain_core.messages import HumanMessage, AIMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

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
    
    # Also update LangGraph state for memory (optional - for checkpointing)
    # chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)