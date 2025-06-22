import gradio as gr

class Chatbot:
    def __init__(self, vector_search_function, llm_function=None):
        """
        Initialize the chatbot with vector search and optional LLM functions.
        :param vector_search_function: Function to perform vector search.
        :param llm_function: Function to get results from an LLM (optional).
        """
        self.vector_search_function = vector_search_function
        self.llm_function = llm_function

    def chat(self, query):
        """
        Handle the chat query and return results from vector search and optionally LLM.
        :param query: User query.
        :return: Tuple of vector search result and LLM result (if provided).
        """
        vector_result = self.vector_search_function(query)
        llm_result = self.llm_function(query) if self.llm_function else "LLM functionality not implemented."
        return vector_result, llm_result

    def launch(self):
        """
        Launch the Gradio interface for the chatbot.
        """
        interface = gr.Interface(
            fn=self.chat,
            inputs=gr.Textbox(label="Enter your query"),
            outputs=[
                gr.Textbox(label="Vector Search Result"),
                gr.Textbox(label="LLM Result")
            ],
            title="Chatbot with Vector Search and LLM",
            description="Enter your query and get results from vector search and optionally LLM."
        )
        interface.launch()