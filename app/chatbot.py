from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

from .chat_wrapper import Samantha, Mistral
from .mistral_retrieval_qa import retrieve_from_str

import requests




class Chatbot:
    def __init__(self):

        self.llm = Ollama(model="mistral-m3allem")
        self.model = Mistral(llm=self.llm, callbacks=[StreamingStdOutCallbackHandler()])
        self.memory = ConversationBufferMemory(return_messages=True)
        

        url = "http://213.171.3.163:1337/api/devices?1=null"
        headers = {
            'Authorization': 'Bearer 76a4b9c3db10dace1d90eaddce9fc7b3bd391a81fd12ea07b71cb9e88571287efb35620a88e0a49d0a6c0977148eb7251d8312dfe9964babfe4a94fc29b93b38f069a6f07e60e928d81af6695309e694bc98b672d0b690951b53388d4f1d5025629f6664df45e437818203bb934d839d47b1353076406c76241bb899a578a52e'
        }

        response = requests.get(url, headers=headers)
        json_data = response.json()
        name = json_data['data'][0]['attributes']['name']
        city = json_data['data'][0]['attributes']['city']
        knowledge = json_data['data'][0]['attributes']['knowledge']
        bannedWords = json_data['data'][0]['attributes']['bannedWords']

        print(name)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Твое имя "+name+" ты туристический гид по городу "+city+"запрещено говорить о "+bannedWords+" разрешено общаться о "+knowledge,
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        self.chat_chain = ConversationChain(
            llm=self.model, prompt=self.prompt, memory=self.memory, verbose=True
        )

    def generate_response(self, user_input):
        """Generates a response based on the user input and context from various sources."""

        data = {"input": user_input}
        return self.chat_chain.invoke(data)
