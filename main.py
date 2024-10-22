import argparse
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
These are the list of professors in Southeastern Louisiana University. The data is formated as: lastname, firstname, title, their degrees and universities that they went to.
These could also be some course requirements for a certain degree. Based on the question provide the answer.
Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    print("\n\n\nLength of results from DB search: ", len(results))
    # print("Results from DB search: ", results, "\n\n\n")
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)
    content = response_text.content

    # model = ChatAnthropic()
    # response_text = model.predict(prompt)

    sources = [
        {
            "source": doc.metadata.get("source", None),
            "page": doc.metadata.get("page", None),
            "score": score
        }
        for doc, score in results
    ]

    formatted_response = f"Response: {content}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()