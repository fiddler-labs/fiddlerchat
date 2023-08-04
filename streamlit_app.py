import pandas as pd
import streamlit as st
import os
import openai
import tiktoken
from scipy import spatial
import ast
import gspread
from google.oauth2 import service_account

# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"
    ],
)
# conn = connect(credentials=credentials)
client = gspread.authorize(credentials)


openai.api_key = os.environ.get('OPENAI_API_KEY')
embeddings_path = "./all_1_8.csv"
df = pd.read_csv(embeddings_path)
# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

CHAT_HISTORY = 'chat_history'
ANSWER = 'answer'
COL_RANGE = 'A:F'
THUMB_UP = "thumbs_up_button"
THUMB_DOWN = "thumbs_down_button"
WHATEVER = "neutral"
COMMENT = "comment"

if "query" not in st.session_state:
    st.session_state["query"] = ""

if "dummy" not in st.session_state:
    st.session_state["dummy"] = "blabla"

if THUMB_DOWN not in st.session_state:
    st.session_state[THUMB_DOWN] = None

if THUMB_UP not in st.session_state:
    st.session_state[THUMB_UP] = None

if WHATEVER not in st.session_state:
    st.session_state[WHATEVER] = None

if CHAT_HISTORY not in st.session_state:
    st.session_state[CHAT_HISTORY] = []

if COMMENT not in st.session_state:
    st.session_state[COMMENT] = ""

if ANSWER not in st.session_state:
    st.session_state[ANSWER] = None





def store_query(
        query: str,
        response: str,
        query_embed,
        response_embed):

    sheet_url = st.secrets["private_gsheets_url"]  # this information should be included in streamlit secret
    sheet = client.open_by_url(sheet_url).get_worksheet(1)
    # existing_data = sheet.get(COL_RANGE)
    # existing_data.append([query, response])
    sheet.append_row([query, response, '', '', str(query_embed), str(response_embed)], table_range=COL_RANGE)
    # st.success('Data has been written to Google Sheets')
    return


def store_feedback(feedback=-1):

    sheet_url = st.secrets["private_gsheets_url"]  # this information should be included in streamlit secret
    sheet = client.open_by_url(sheet_url).get_worksheet(1)
    q_list = sheet.col_values(2)
    rows = len(q_list)
    # print(st.session_state.query)
    if q_list[rows-1] == st.session_state[ANSWER]:
        sheet.update(f'C{rows}', feedback)
    # st.success('Data has been written to Google Sheets')
    return

def store_comment():

    sheet_url = st.secrets["private_gsheets_url"]  # this information should be included in streamlit secret
    sheet = client.open_by_url(sheet_url).get_worksheet(1)
    q_list = sheet.col_values(2)
    rows = len(q_list)
    # print(st.session_state.query)
    if q_list[rows-1] == st.session_state[ANSWER]:
        sheet.update(f'D{rows}', st.session_state[COMMENT])
    st.success('Your comment is noted!')
    return



# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
):
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n], query_embedding


# models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int,
    introduction='You are a tool called Fiddler Chatbot and your purpose is to use the below documentation from the company Fiddler to answer the subsequent documentation questions. Also, if possible, give me the reference URLs according to the following instructions. The way to create the URLs is: if you are discussing a client method or an API reference add "https://docs.fiddler.ai/reference/" before the "slug" value of the document. If it is Guide documentation add "https://docs.fiddler.ai/docs/" before before the "slug" value of the document. Only use the value following "slug:" to create the URLs and do not use page titles for slugs. If you are using quickstart notebooks, do not generate references. Note that if a user asks about uploading events, it means the same as publishing events. If the answer cannot be found in the documentation, write "I could not find an answer."'

):
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses, query_embed = strings_ranked_by_relatedness(query, df)
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = string
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question, query_embed


def ask(
    # query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
    temperature: int = 0,
    # chat_history=None,
    introduction='You are a tool called Fiddler Chatbot and your purpose is to use the below documentation from the company Fiddler to answer the subsequent documentation questions. Also, if possible, give me the reference URLs according to the following instructions. The way to create the URLs is: if you are discussing a client method or an API reference add "https://docs.fiddler.ai/reference/" before the "slug" value of the document. If it is Guide documentation add "https://docs.fiddler.ai/docs/" before before the "slug" value of the document. Only use the value following "slug:" to create the URLs and do not use page titles for slugs. If you are using quickstart notebooks, do not generate references. Note that if a user asks about uploading events, it means the same as publishing events. If the answer cannot be found in the documentation, write "I could not find an answer."'

):
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    if st.session_state.query:
        query = st.session_state.query
        chat_history = st.session_state[CHAT_HISTORY]
        if chat_history is None:
            chat_history = []
        chat_history.append("User Query: "+query)
        # query = "\n".join(chat_history)
        message, query_embed = query_message(query, df=df, model=model, token_budget=token_budget, introduction = introduction)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "You answer questions about Fiddler documentation."},
            {"role": "user", "content": message},
        ]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        response_message = response["choices"][0]["message"]["content"]
        response_embedding_response = openai.Embedding.create(
            model=EMBEDDING_MODEL,
            input=response_message,
        )
        response_embed = response_embedding_response["data"][0]["embedding"]
        chat_history.append("Bot response: "+response_message)
        st.session_state[ANSWER], st.session_state[CHAT_HISTORY] = response_message, chat_history
        store_query(query, response_message, query_embed, response_embed)
    return


# def on_enter_pressed():
#     st.write("Enter key pressed! Entered text:", st.session_state.dummy)
# Streamlit app
def main():
    st.image('poweredby.jpg', width=550)
    st.title("Fiddler Chatbot")

    # User input
    st.text_input("Your Question:", key="query", on_change=ask)

    if st.button("Reset Chat History"):
        st.session_state[CHAT_HISTORY] = []
        st.session_state[ANSWER] = None

    if st.session_state[ANSWER] is not None:
        st.text("Bot:")
        st.write(st.session_state[ANSWER])
        st.text_input("Any comments on the bot response?", key="comment", on_change=store_comment, value="")
        # Display thumbs up and thumbs down buttons
        col1, col2, col3 = st.columns([0.5, 0.5, 5])
        with col1:
            if not st.session_state[THUMB_UP] or st.session_state[THUMB_UP] is None:
                st.button("👍", key="thumbs_up_button", on_click=store_feedback, kwargs={'feedback':1})
        with col2:
            if not st.session_state[THUMB_DOWN] or st.session_state[THUMB_DOWN] is None:
                st.button("👎", key="thumbs_down_button", on_click=store_feedback, kwargs={'feedback':0})
        with col3:
            if not st.session_state[WHATEVER] or st.session_state[WHATEVER] is None:
                st.button("🤷", key="neutral", on_click=store_feedback)
            # User input

    with st.container():
        st.header("Chat History")
        st.write("\n\n".join(st.session_state[CHAT_HISTORY]))



if __name__ == "__main__":
    main()


