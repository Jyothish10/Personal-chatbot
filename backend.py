import os
from langchain_community.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


def demo_chatbot():
    demo_llm = Bedrock(
        credentials_profile_name='default',
        model_id="amazon.titan-text-lite-v1",
        model_kwargs={
            "maxTokenCount": 200,
            "temperature": 0.7,
            "topP": 0.9
        }
    )

    return demo_llm


    '''return demo_llm.predict(input_text)
response=demo_chatbot("what is your name?")
print(response)'''

def demo_memory():
    llm_data=demo_chatbot()
    memory=ConversationBufferMemory(
        llm=llm_data,
        max_token_limit=512
    )
    return memory


def demo_conversation(input_text, memory):
    llm_chain_data = demo_chatbot()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)

    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply
