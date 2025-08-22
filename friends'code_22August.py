import os
import json
import time
import logging
import argparse
import subprocess
import re
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, Field
import traceback
from PIL import Image
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from io import StringIO
import threading
import fitz  # PyMuPDF
from PIL import Image
import shelve

# Vector Database and Embedding
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# LLM and Chain Components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.schema import StrOutputParser
import google.generativeai as genai
from langchain_experimental.agents import create_pandas_dataframe_agent

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import FastEmbedEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sqlite3

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Local storage configuration
LOCAL_STORAGE_ROOT = "local_storage"
os.makedirs(LOCAL_STORAGE_ROOT, exist_ok=True)

# Subdirectories
EMBEDDINGS_DIR = os.path.join(LOCAL_STORAGE_ROOT, "embeddings")
CACHE_DIR = os.path.join(LOCAL_STORAGE_ROOT, "cache")
INVOICES_DIR = os.path.join(LOCAL_STORAGE_ROOT, "invoices")
EXTRACTS_DIR = os.path.join(LOCAL_STORAGE_ROOT, "extracts")


for dir_path in [EMBEDDINGS_DIR, CACHE_DIR, INVOICES_DIR, EXTRACTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Configuration
DB_PATH = os.path.join(EXTRACTS_DIR, "invoices.db")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")
# TARGET_PHONE_NUMBER = os.getenv("TARGET_PHONE_NUMBER")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CACHE_EXPIRY = 3600  # 1 hour
SIGNAL_CLI_PATH = "/usr/local/bin/signal-cli"

# Collection name for invoice data
INVOICE_COLLECTION_NAME = "invoice_embeddings"

# Define the embedding model name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # This is a good balance of speed and quality
ATTACHMENT_DIR = os.getenv("ATTACHMENT_DIR", "uploads")
os.makedirs(ATTACHMENT_DIR, exist_ok=True)
genai.configure(api_key=GOOGLE_API_KEY)


class LocalCache:
    def __init__(self, db_name="cache"):
        self.cache_file = os.path.join(CACHE_DIR, f"{db_name}.db")

    def get(self, key):
        with shelve.open(self.cache_file) as db:
            return db.get(key)

    def set(self, key, value, expiry=None):
        with shelve.open(self.cache_file) as db:
            db[key] = value

    def keys(self, pattern):
        with shelve.open(self.cache_file) as db:
            return [k for k in db.keys() if re.match(pattern, k)]


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        """Initialize with SentenceTransformer model."""
        self.model = SentenceTransformer(model_name)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        """Embed search documents."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        """Embed query text."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def __call__(self, text):
        if isinstance(text, list):
            return self.embed_documents(text)
        else:
            return self.embed_query(text)


class InvoiceData(BaseModel):
    Is_Invoice: bool = Field(
        ..., description="Indicates whether it is an invoice or not"
    )
    Invoice_Number: str = Field(
        ..., description="Invoice number extracted from the document"
    )
    Invoice_Date: str = Field(..., description="Date of the invoice")
    Net_SUM: str = Field(..., description="Net total amount before VAT")
    Gross_SUM: str = Field(..., description="Gross total including VAT")
    VAT_Percentage: str = Field(..., description="VAT percentage")
    VAT_Amount: str = Field(..., description="VAT amount in invoice currency (EUR)")
    Invoice_Sender_Name: str = Field(..., description="Sender Name")
    Invoice_Sender_Address: str = Field(..., description="Sender Address")
    Invoice_Recipient_Name: str = Field(..., description="Recipient Name")
    Invoice_Recipient_Address: str = Field(..., description="Recipient Address")
    Invoice_Payment_Terms: str = Field(None, description="Payment terms (e.g., NET 30)")
    Payment_Method: str = Field(None, description="Payment method used for the invoice")
    Category_Classification: str = Field(
        None, description="Bookkeeping category (e.g., SOFTWARE, Electronics)"
    )
    Is_Subscription: bool = Field(
        ..., description="Indicates whether the invoice is for a subscription"
    )
    START_Date: str = Field(
        None,
        description="Subscription start date, applicable only if is_Subscription is True",
    )
    END_Date: str = Field(
        None,
        description="Subscription end date, applicable only if is_Subscription is True",
    )
    Tips: str = Field(None, description="If any tips mentioned in the invoice")
    Original_Filename: str = Field(None, description="Original filename of the invoice")
    Upload_Timestamp: str = Field(
        None, description="Timestamp when invoice was uploaded"
    )
    Handwritten_Comment: str = Field(
        None, description="If any handwritten comment is mentioned in the invoice"
    )
    Model_Abouts: Optional[Dict[str, str]] = Field(
        None,
        description="Stores model token usage metadata like Input_Tokens, Output_Tokens, Total_Tokens",
    )


def store_uploaded_file(file_path: str) -> str:
    """Store uploaded file in local invoices directory"""
    try:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(INVOICES_DIR, filename)

        # Handle duplicate filenames
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            dest_path = os.path.join(INVOICES_DIR, f"{name}_{counter}{ext}")
            counter += 1

        shutil.copy(file_path, dest_path)
        return dest_path
    except Exception as e:
        logger.error(f"Error storing file: {str(e)}")
        return None


# def save_results_to_sheet(results: List[Dict[str, Any]]) -> str:
#     """
#     Save/update invoice data in a single local Excel file (extractedData.xlsx),
#     but split into month-wise sheets based on Invoice_Date.
#     Returns path to the saved file
#     """
#     try:
#         # Create DataFrame from new results
#         new_df = pd.DataFrame(results)

#         # Ensure Invoice_Date is in datetime format
#         if "Invoice_Date" not in new_df.columns:
#             raise ValueError("Missing 'Invoice_Date' column in results")
#         new_df["Invoice_Date"] = pd.to_datetime(new_df["Invoice_Date"], errors="coerce")

#         # Handle nested dictionaries (like Model_Abouts)
#         for col in new_df.columns:
#             if new_df[col].apply(lambda x: isinstance(x, dict)).any():
#                 new_df[col] = new_df[col].apply(json.dumps)

#         # Create directory if it doesn't exist
#         os.makedirs(EXTRACTS_DIR, exist_ok=True)

#         # Define the single output file path
#         output_path = os.path.join(EXTRACTS_DIR, "extractedData.xlsx")

#         # Dictionary to hold all sheet DataFrames
#         sheet_data = {}

#         # If file exists, load existing sheets
#         if os.path.exists(output_path):
#             try:
#                 existing_sheets = pd.read_excel(output_path, sheet_name=None)  # Load all sheets
#                 sheet_data.update(existing_sheets)
#             except Exception as e:
#                 logger.warning(f"Error reading existing file, creating new: {str(e)}")
#                 sheet_data = {}

#         # Group new data by month
#         new_df["Month_Sheet"] = new_df["Invoice_Date"].dt.strftime("%B_%Y").str.lower()
#         for sheet_name, group in new_df.groupby("Month_Sheet"):
#             if sheet_name in sheet_data:
#                 # Append to existing sheet
#                 combined_df = pd.concat([sheet_data[sheet_name], group], ignore_index=True)
#                 # Remove duplicates based on Invoice_Number if exists
#                 if "Invoice_Number" in combined_df.columns:
#                     combined_df = combined_df.drop_duplicates(
#                         subset=["Invoice_Number"], keep="last"
#                     )
#                 sheet_data[sheet_name] = combined_df
#             else:
#                 sheet_data[sheet_name] = group

#         # Save all sheets to Excel
#         with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
#             for sheet_name, df in sheet_data.items():
#                 df.to_excel(writer, index=False, sheet_name=sheet_name)

#             # Add/update metadata sheet
#             total_records = sum(len(df) for df in sheet_data.values())
#             metadata = {
#                 "Last_Updated": [datetime.now().isoformat()],
#                 "Total_Records": [total_records],
#                 "Sheets": [", ".join(sheet_data.keys())],
#                 "New_Records_Added": [len(new_df)],
#             }
#             pd.DataFrame(metadata).to_excel(writer, index=False, sheet_name="Metadata")

#         logger.info(
#             f"Updated {output_path} with {len(new_df)} new records split into {len(sheet_data)} month-wise sheets"
#         )
#         return output_path

#     except Exception as e:
#         logger.error(f"Error saving results to Excel: {str(e)}")
#         traceback.print_exc()
#         return None

def save_results_to_sheet(results: List[Dict[str, Any]]) -> str:
    """
    Save/update invoice data in a single local Excel file (extractedData.xlsx),
    but split into month-wise sheets based on Invoice_Date.
    Returns path to the saved file
    """
    try:
        # Create DataFrame from new results
        new_df = pd.DataFrame(results)

        # Ensure Invoice_Date is in datetime format
        if "Invoice_Date" not in new_df.columns:
            raise ValueError("Missing 'Invoice_Date' column in results")
        new_df["Invoice_Date"] = pd.to_datetime(new_df["Invoice_Date"], errors="coerce")

        # Handle nested dictionaries (like Model_Abouts)
        for col in new_df.columns:
            if new_df[col].apply(lambda x: isinstance(x, dict)).any():
                new_df[col] = new_df[col].apply(json.dumps)

        # Create directory if it doesn't exist
        os.makedirs(EXTRACTS_DIR, exist_ok=True)

        # Define the single output file path
        output_path = os.path.join(EXTRACTS_DIR, "extractedData.xlsx")

        # Dictionary to hold all sheet DataFrames
        sheet_data = {}

        # If file exists, load existing sheets
        if os.path.exists(output_path):
            try:
                existing_sheets = pd.read_excel(output_path, sheet_name=None)  # Load all sheets
                sheet_data.update(existing_sheets)
            except Exception as e:
                sheet_data = {}

        # Group new data by month
        new_df["Month_Sheet"] = new_df["Invoice_Date"].dt.strftime("%B_%Y").str.lower()
        for sheet_name, group in new_df.groupby("Month_Sheet"):
            if sheet_name in sheet_data:
                # Append to existing sheet
                combined_df = pd.concat([sheet_data[sheet_name], group], ignore_index=True)
                # Remove duplicates based on Invoice_Number if exists
                if "Invoice_Number" in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(
                        subset=["Invoice_Number"], keep="last"
                    )
                sheet_data[sheet_name] = combined_df
            else:
                sheet_data[sheet_name] = group

        # Save all sheets to Excel
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in sheet_data.items():
                df.to_excel(writer, index=False, sheet_name=sheet_name)

            # Add/update metadata sheet
            total_records = sum(len(df) for df in sheet_data.values())
            metadata = {
                "Last_Updated": [datetime.now().isoformat()],
                "Total_Records": [total_records],
                "Sheets": [", ".join(sheet_data.keys())],
                "New_Records_Added": [len(new_df)],
            }
            pd.DataFrame(metadata).to_excel(writer, index=False, sheet_name="Metadata")
        
        return output_path

    except Exception as e:
        return None


def save_invoices_to_sqlite(df: pd.DataFrame):
    """
    Save invoice dataframe into SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS invoices (
        Invoice_Number TEXT,
        Invoice_Date TEXT,
        Net_SUM REAL,
        Gross_SUM REAL,
        VAT_Percentage REAL,
        VAT_Amount REAL,
        Invoice_Sender_Name TEXT,
        Invoice_Sender_Address TEXT,
        Invoice_Recipient_Name TEXT,
        Invoice_Recipient_Address TEXT,
        Invoice_Payment_Terms TEXT,
        Payment_Method TEXT,
        Category_Classification TEXT,
        Is_Subscription TEXT,
        START_Date TEXT,
        END_Date TEXT,
        Tips TEXT,
        Handwritten_Comment TEXT,
        Model_Abouts TEXT
    )
    """)

    # Insert or replace (avoids duplicates on same Invoice_Number)
    for _, row in df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO invoices VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            row['Invoice_Number'],
            row['Invoice_Date'],
            row['Net_SUM'],
            row['Gross_SUM'],
            row['VAT_Percentage'],
            row['VAT_Amount'],
            row['Invoice_Sender_Name'],
            row['Invoice_Sender_Address'],
            row['Invoice_Recipient_Name'],
            row['Invoice_Recipient_Address'],
            row['Invoice_Payment_Terms'],
            row['Payment_Method'],
            row['Category_Classification'],
            row['Is_Subscription'],
            row['START_Date'],
            row['END_Date'],
            row['Tips'],
            row['Handwritten_Comment'],
            str(row['Model_Abouts'])  # ensure JSON/dict converted to string
        ))

    conn.commit()
    conn.close()
    logger.info("✅ Invoices saved to SQLite database at %s", DB_PATH)


def embed_and_upload_invoice_data():
    try:
        # Load the most recent extract
        file_path = os.path.join(EXTRACTS_DIR, "extractedData.xlsx")

        if not os.path.exists(file_path):
            response = "❌ No invoice data found in local database."
            return response
        
        df = pd.read_excel(file_path, sheet_name="Invoices")
        save_invoices_to_sqlite(df)
        markdown_rows = []
        for _, row in df.iterrows():
            row_markdown = f"""
            ### Invoice Record
            - Invoice Number: {row['Invoice_Number']}
            - Invoice Date: {row['Invoice_Date']}
            - Net SUM: {row['Net_SUM']}
            - Gross SUM: {row['Gross_SUM']}
            - VAT %: {row['VAT_Percentage']}
            - VAT Amount: {row['VAT_Amount']}
            - Sender: {row['Invoice_Sender_Name']}
            - Sender Address: {row['Invoice_Sender_Address']}
            - Recipient: {row['Invoice_Recipient_Name']}
            - Recipient Address: {row['Invoice_Recipient_Address']}
            - Payment Terms: {row['Invoice_Payment_Terms']}
            - Payment Method: {row['Payment_Method']}
            - Category: {row['Category_Classification']}
            - Is Subscription: {row['Is_Subscription']}
            - Start Date: {row['START_Date']}
            - End Date: {row['END_Date']}
            - Tips: {row['Tips']}
            - Handwritten Comment: {row['Handwritten_Comment']}
            - Model About: {row['Model_Abouts']}
            """
            markdown_rows.append(row_markdown.strip())

        docs = [Document(page_content=row, metadata={"source": file_path}) for row in markdown_rows]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        # ✅ Setup vector DB
        embedding = FastEmbedEmbeddings()
        db = Chroma(
            embedding_function=embedding,
            collection_name="invoices",
            persist_directory="invoice_store"
        )
        client = db._client  # chromadb.Client
        collections = [c.name for c in client.list_collections()]
        if "invoices" in collections:
            logger.info("Deleting existing 'invoices' collection before re-embedding")
            client.delete_collection("invoices")

        db.delete_collection()
        
        db.add_documents(chunks)
        logger.info(f"Successfully embedded and uploaded {len(texts)} invoice records")
        return True
    except Exception as e:
        logger.error(f"Error embedding and uploading invoice data: {str(e)}")
        return False


def analyze_invoice(image_path: str, filename: str, context: str = "") -> InvoiceData:
    try:
        img = Image.open(image_path)

        context_section = f"\nConversation Context:\n{context}" if context else ""

        system_prompt = f"""
            You are a multilingual invoice parser. {context_section}
            
            Analyze OCR and image data in Serbian (Cyrillic/Latin), Bulgarian, English, French, or German. Extract structured fields without guessing or fabricating.

            **General Rules:**
            - Use commas as decimal separators.
            - Format VAT and percentages like 00.00%.
            - Omit currency symbols (€, RSD).
            - If a field is missing, return "N/A".
            - Use both OCR and visual cues.
            - Use true/false do not use yes/no.
            - All key pair will be in string format do not use integer.

            **Mandatory Fields:**
            {{
                "Is_Invoice": "Indicates whether this is an invoice or not.",
                "Invoice_Number": "Invoice number extracted from the document",
                "Invoice_Date": "Date of the invoice",
                "Gross_SUM": "Mentioned in the invoice including VAT Amount nearly paid amount",
                "Paid_Amount": "Amount tendered/provided by customer",
                "Change_Amount": "If applicable, amount returned to customer",
                "Net_SUM": "if not mentioned in the invoice calculate it Gross SUM - VAT Amount",
                "VAT_Percentage": "VAT percentage",
                "VAT_Amount": "VAT amount in invoice currency (EUR)",
                "Invoice_Sender_Name": "Sender Name",
                "Invoice_Sender_Address": "Sender Address",
                "Invoice_Recipient_Name": "Recipient Name",
                "Invoice_Recipient_Address": "Recipient Address",
                "Invoice_Payment_Terms": "Payment terms (e.g., NET 30)",
                "Category_Classification": "Categorize as it would be in Austrian bookkeeping (e.g., SOFTWARE, Electronics, Food & Beverage, Petrol Pump, etc.)",
                "Payment_Method": "Payment method used for the invoice",
                "Is_Subscription": "Indicates whether the invoice is for a subscription",
                "START_Date": "Subscription start date (if applicable, otherwise N/A)",
                "END_Date": "Subscription end date (if applicable, otherwise N/A)",
                "Tips": "Any tips mentioned in the invoice",
                "Handwritten_Comment": "Any handwritten content in the invoice"
            }}
            **Calculations:**
            - Change_Amount = Paid_Amount - Gross_SUM (if Paid_Amount > Gross_SUM).
            - Net_SUM = Gross_SUM - VAT_Amount (if missing).
            - Tips = Paid_Amount - Gross_SUM (if overpaid).

            **Validation Rules:**
            - Gross_SUM >= Net_SUM.
            - Only accept invoice numbers labeled like "Invoice No.", "Invoice Number", "Faktura br", etc.
            - Ignore random long numbers (PIB, OIB, IBAN) unless labeled as invoice numbers.
            - If Recipient is missing, set as "Cash Customer" and Address as "Unknown".

            **Extra:**
            - Prioritize handwritten notes if found.
            - Match values near labels, considering multi-language layouts.
            - Return a valid, clean JSON object. Do NOT include markdown formatting, do not use tags like ```json in the response.

            Follow the guidelines strictly.
            """

        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(
            [
                system_prompt,
                "Analyze this invoice image and extract all relevant information following the guidelines. Respond in English with JSON only.",
                img,
            ]
        )

        # Extract JSON from response
        response_text = response.text
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"(\{[\s\S]*\})", response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise ValueError("Could not extract JSON from LLM response")

        # Parse JSON
        extracted_data = json.loads(json_str)
        pretty_response_str = "\n".join(
            [f"{key}: {value}" for key, value in extracted_data.items()]
        )

        try:
            output_tokens = response.usage_metadata.candidates_token_count
            input_tokens = response.usage_metadata.prompt_token_count
            total_tokens = response.usage_metadata.total_token_count
        except AttributeError:
            output_tokens = 0
            input_tokens = 0
            total_tokens = 0

        # Add additional metadata
        extracted_data["Original_Filename"] = filename
        extracted_data["Upload_Timestamp"] = datetime.now().isoformat()
        extracted_data["Model_Abouts"] = {
            "Input_Tokens": str(input_tokens),
            "Output_Tokens": str(output_tokens),
            "Total_Tokens": str(total_tokens),
        }

        # Convert to InvoiceData model
        invoice_data = InvoiceData(**extracted_data)

        return pretty_response_str, extracted_data, invoice_data

    except Exception as e:
        logger.error(f"Error analyzing invoice: {str(e)}")
        raise

# Router prompt for determining type of query
def create_router_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    router_prompt = PromptTemplate.from_template(
        """
    Based on the user's question and conversation history, determine whether this is a greeting or an invoice-related query.
    
    History:
    {history}
    
    - If the question is a greeting (hello, hi, good morning, etc.) or general conversation, respond with `BASE_AGENT`.
    - If the question is related to invoice data, invoice processing, or any specific invoice information, or related to previous query, respond with `RAG_AGENT`.
    - If the question asks to delete an invoice (example: "delete invoice number 12345" or "remove invoice 98765"), respond with `DELETE_AGENT`.
    - If the question is about summary, expense analysis between specific dates, or spending in categories like travel, groceries, flights, etc., respond with `TRIP_ANALYZER_AGENT`.
    - If the question is unclear or unrelated to invoices, respond with `OTHER`.
    
    Question: {question}
    """
    )

    # Create router chain
    router_chain = router_prompt | llm | StrOutputParser()
    return router_chain


def pdf_to_image(pdf_path, page_num=0, output_path="page.jpg"):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        pix.save(output_path)
        return output_path
    except Exception as e:
        raise Exception("Error converting PDF to image")


def create_excel_agent(full_prompt: str) -> str:
    try:
        
        embedding = FastEmbedEmbeddings()
        db = Chroma(
            embedding_function=embedding,
            collection_name="invoices",
            persist_directory="invoice_store"
        )

        retriever = db.as_retriever(search_kwargs={"k": 10})

        # ✅ Setup LLM
        llm = ChatOpenAI(
            model="llama3-70b-8192",
            temperature=0.2,
            top_p=0.9,
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1"
        )

        prompt = PromptTemplate(
            template=("""
                You are a helpful invoice assistant.
                Carefully check all the invoices details while responding.
                Most Important: Do not modify invoice numbers (keep exact format, 
                even if they include special characters like / or П).
                      
                Context:
                {context}

                Question:
                {question}

                Answer:
            """),
            input_variables=["context", "question"],
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        # ✅ Run the query
        agent_result = qa.invoke({"query": f"Answer this query about invoices: {full_prompt}."})
        return agent_result["result"]

    except Exception as e:
        logger.error(f"Error during RAG query: {str(e)}")
        return "❌ Error accessing invoice data. Please try again."
    

def create_focused_query(full_prompt: str, query_params: dict) -> str:
    return f"""
    You are a professional Trip Expense Analysis Assistant.

    The user has asked: "{full_prompt}"

    You are provided with a set of structured invoice rows (as markdown tables).
    Each row contains details like invoice date, sender name, amount, category, and currency.

    Perform the following:
    1. Filter the rows based on date or category if applicable.
    2. Group rows by `Category_Classification`.
    3. For each group, calculate:
    - Total amount using `Gross_SUM` column.
    - List of `Invoice_Sender_Name` and `Invoice_Sender_Address` for trip origin.
    4. Convert all non-RSD currencies to **RSD** using:
    - USD 1 = RSD 102.96
    - INR 1 = RSD 1.21
    - AED 1 = RSD 28.03
    5. Format the final result as a markdown table with the following headers:
    | Category_Classification | Invoice_Date | Invoice_Sender_Name | Invoice_Sender_Address | Gross_SUM_RSD |

    Important:
    - Do not generate fake rows — only summarize what is given in the context.
    - If no rows match, respond with: "No invoices found."
    - Do not wrap the output in triple backticks or language tags.
    - Ensure the table is clean, correctly aligned, and uses float values for totals.

    Be precise and concise. Only respond with the structured summary table.
    """


def markdown_to_df(markdown: str) -> pd.DataFrame:
    try:
        lines = markdown.strip().split("\n")

        # Get only lines that look like table rows
        table_lines = [line for line in lines if line.strip().startswith("|")]

        # Remove separator (---) row
        table_lines = [
            line for line in table_lines if not re.match(r"^\|\s*-", line.strip())
        ]

        if not table_lines or len(table_lines) < 2:
            return None  # Need header + at least 1 row

        # Convert to TSV-style string safely (to avoid comma confusion)
        tsv_string = "\n".join(
            [
                "\t".join([cell.strip() for cell in row.strip("|").split("|")])
                for row in table_lines
            ]
        )

        # Read into DataFrame using tab separator
        df = pd.read_csv(StringIO(tsv_string), sep="\t")

        return df
    except Exception as e:
        print(f"❌ Error converting markdown to DataFrame: {str(e)}")
        return None


def df_to_trip_png(df: pd.DataFrame, output_folder="trip_images") -> str:
    try:
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trip_summary_{timestamp}.png"
        file_path = os.path.join(output_folder, filename)

        fig, ax = plt.subplots(
            figsize=(min(20, len(df.columns) * 2), len(df) * 0.6 + 1)
        )
        ax.axis("off")
        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.tight_layout()
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        return file_path

    except Exception as e:
        print(f"Error generating PNG from DataFrame: {str(e)}")
        return None

def create_pandas_agent(df):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_execution_time=1,
    )
    agent = create_pandas_dataframe_agent(
        llm, df, handle_parsing_errors=True, verbose=True, allow_dangerous_code=True
    )
    return agent

class SignalBot:
    def __init__(self, phone_number: str):
        self.phone_number = phone_number
        self.event_db = LocalCache("events")  # Local cache for events
        self.cache_db = LocalCache("messages")  # Local cache for messages

        # Initialize router chain
        self.router_chain = create_router_chain()

        logger.info(f"Signal Bot initialized with phone number: {phone_number}")

    def process_incoming_message(
        self,
        sender: str,
        message_text: str,
        timestamp: int,
        attachments: List[str] = None,
    ) -> str:
        logger.info(f"Processing message from {sender}: {message_text}")
        has_attachment = attachments is not None and len(attachments) > 0
        has_message = message_text.strip() != ""

        try:
            # Store the incoming message event
            event_data = {
                "sender": sender,
                "message": message_text,
                "timestamp": timestamp,
                "type": "incoming",
                "attachments": attachments if has_attachment else [],
            }
            event_key = f"event:{timestamp}:{sender}"
            self.event_db.set(event_key, json.dumps(event_data))

            if has_attachment and len(attachments) > 1:
                response = "📁 I received multiple files. I'll process them one by one with a brief pause between each to avoid rate limits."
                self.send_message(sender, response)

                def process_batch_attachments():
                    try:
                        successful_count = 0
                        for i, attachment_path in enumerate(attachments):
                            logger.info(
                                f"Processing attachment {i+1}/{len(attachments)}"
                            )

                            # Convert PDF to image if needed
                            if attachment_path.lower().endswith(".pdf"):
                                file_path = pdf_to_image(attachment_path)
                            else:
                                file_path = attachment_path

                            filename = os.path.basename(file_path)
                            stored_path = store_uploaded_file(file_path)

                            # Process the invoice
                            try:
                                _, extracted_data, invoice_data = analyze_invoice(
                                    file_path, filename, context=""
                                )
                                response = (
                                    "Here's what I found in the invoice:\n\n"
                                    + "\n".join(
                                        [f"{k}: {v}" for k, v in extracted_data.items()]
                                    )
                                )

                                if extracted_data and extracted_data.get(
                                    "Is_Invoice", False
                                ):
                                    invoice_data.Model_Abouts["Local_Path"] = (
                                        stored_path
                                    )
                                    save_results_to_sheet([invoice_data.model_dump()])
                                    embed_and_upload_invoice_data()
                                    self.send_message(sender, response)
                                    response += (
                                        "\n\n✅ Invoice saved to local database!"
                                    )
                                    # Send progress update
                                    progress_msg = f"✅ Processed {i+1}/{len(attachments)}: {filename}"
                                    if extracted_data.get("Invoice_Number"):
                                        progress_msg += f" (Invoice #{extracted_data['Invoice_Number']})"
                                    self.send_message(sender, progress_msg)
                                    successful_count += 1
                                else:
                                    self.send_message(
                                        sender,
                                        f"❌ {filename} doesn't appear to be a valid invoice.",
                                    )

                            except Exception as e:
                                logger.error(f"Error processing {filename}: {str(e)}")
                                self.send_message(
                                    sender, f"❌ Failed to process {filename}: {str(e)}"
                                )

                            # Wait 10 seconds between processing to avoid rate limits
                            if (
                                i < len(attachments) - 1
                            ):  # Don't wait after the last one
                                time.sleep(10)

                        # Final completion message
                        if successful_count > 0:
                            self.send_message(
                                sender,
                                f"🎉 Finished processing {successful_count}/{len(attachments)} files successfully!",
                            )
                            # Update embeddings after batch processing
                            embed_and_upload_invoice_data()
                        else:
                            self.send_message(
                                sender,
                                "❌ No valid invoices were processed from the batch.",
                            )

                    except Exception as e:
                        logger.error(f"Batch processing error: {str(e)}")
                        self.send_message(
                            sender,
                            "❌ Error during batch processing. Some files may not have been processed.",
                        )

                # Start batch processing in background
                threading.Thread(target=process_batch_attachments, daemon=True).start()
                return response

            if has_attachment and attachments[0].lower().endswith(".pdf"):
                file_path = pdf_to_image(attachments[0])
                logger.info(f"PDF Path is {file_path}")
            else:
                file_path = attachments[0] if has_attachment else None

            logger.info(f"File Path is {file_path}")

            # Check cache for this message
            response = None
            if not has_attachment:
                cache_key = f"cache:{sender}:{message_text}"
                cached_response = self.cache_db.get(cache_key)
                if cached_response:
                    logger.info(f"Cache hit for message: {message_text}")
                    response = json.loads(cached_response)["response"]
                    self.send_message(sender, response)
                    logger.info(f"Retrieved cached response: {response}")
                    return response

            # Always get the conversation history first
            history = self._get_conversation_history(sender, 3)
            context = f"Previous conversation: {history}\n\n" if history else ""
            full_prompt = f"{context}User: {message_text}"

            # Determine the type of query using the router chain
            if has_attachment and has_message:
                logger.info("Processing both attachment and message text")
                router_output = "FILE_MESSAGE_AGENT"
            elif has_attachment:
                router_output = "FILE_AGENT"
            else:
                router_query = {"question": message_text, "history": history}
                router_output_raw = self.router_chain.invoke(router_query)
                router_output = router_output_raw.strip().strip("`").upper()

            logger.info(f"Router output: {router_output}")

            if router_output == "FILE_MESSAGE_AGENT":
                logger.info("Using FILE_MESSAGE agent to process attachment with query")
                filename = os.path.basename(file_path)
                stored_path = store_uploaded_file(file_path)

                # First answer the user's question
                img = Image.open(file_path)
                model = genai.GenerativeModel("gemini-2.5-pro")
                query_response = model.generate_content(
                    [
                        f"You are an invoice analysis assistant. The user has sent you an invoice image with this question: '{message_text}'. Answer their question directly based on what you can see in the invoice.",
                        img,
                    ]
                )

                response = f"Based on your invoice image: {query_response.text}\n\nI'm processing this invoice for your records - I'll notify you when complete."
                self.send_message(sender, response)

                def process_invoice_in_background():
                    try:
                        _, extracted_data, invoice_data = analyze_invoice(
                            file_path, filename, context=context
                        )

                        print(
                            "----------------------------------------",
                            extracted_data and extracted_data.get("Is_Invoice", False),
                        )
                        print(
                            extracted_data,
                            "+++++++++++++++++++++++++++++++++++",
                            extracted_data.get("Is_Invoice"),
                        )
                        if extracted_data and extracted_data.get("Is_Invoice", False):
                            # Store locally instead of Google Drive
                            invoice_data.Model_Abouts["Local_Path"] = stored_path
                            save_results_to_sheet([invoice_data.model_dump()])
                            embed_and_upload_invoice_data()
                            self.send_message(
                                sender,
                                "✅ Invoice processed and saved to local database!",
                            )
                        else:
                            self.send_message(
                                sender, "❌ This doesn't appear to be a valid invoice."
                            )

                    except Exception as e:
                        logger.error(f"Background processing error: {str(e)}")
                        self.send_message(
                            sender, "❌ Error processing invoice. Please try again."
                        )

                threading.Thread(
                    target=process_invoice_in_background, daemon=True
                ).start()
                return response

            elif router_output == "FILE_AGENT":
                logger.info("Using FILE agent to process attachment")
                if attachments:
                    filename = os.path.basename(file_path)
                    stored_path = store_uploaded_file(file_path)

                    self.send_message(sender, "We are working on it...")

                    _, extracted_data, invoice_data = analyze_invoice(
                        file_path, filename, context=context
                    )

                    response = "Here's what I found in the invoice:\n\n" + "\n".join(
                        [f"{k}: {v}" for k, v in extracted_data.items()]
                    )

                    if extracted_data.get("Is_Invoice", False):
                        invoice_data.Model_Abouts["Local_Path"] = stored_path
                        save_results_to_sheet([invoice_data.model_dump()])
                        embed_and_upload_invoice_data()
                        response += "\n\n✅ Invoice saved to local database!"
                    else:
                        response += "\n\n❌ This doesn't appear to be a valid invoice."

                    self.send_message(sender, response)
                else:
                    response = "You mentioned a file but I didn't receive one. Please try again."
                    self.send_message(sender, response)

            elif router_output == "RAG_AGENT":
                logger.info("Using RAG agent to process query")
                try:
                    # Load local invoice data
                    response = create_excel_agent(full_prompt)
                    self.send_message(sender, response)

                except Exception as e:
                    logger.error(f"Error during RAG query: {str(e)}")
                    response = "❌ Error accessing invoice data. Please try again."
                    self.send_message(sender, response)

            elif router_output == "TRIP_ANALYZER_AGENT":
                logger.info("Using TRIP_ANALYZER agent")
                try:
                    # Similar to RAG_AGENT but with trip-specific processing
                    extract_files = [
                        f
                        for f in os.listdir(EXTRACTS_DIR)
                        if f.startswith("invoice_results_") and f.endswith(".xlsx")
                    ]
                    if not extract_files:
                        response = "❌ No invoice data found for trip analysis."
                        self.send_message(sender, response)
                        return response

                    latest_file = max(
                        extract_files,
                        key=lambda f: os.path.getmtime(os.path.join(EXTRACTS_DIR, f)),
                    )
                    df = pd.read_excel(os.path.join(EXTRACTS_DIR, latest_file))

                    agent_query = create_focused_query(full_prompt, {})
                    pandas_agent = create_pandas_agent(df)
                    agent_result = pandas_agent.run(agent_query)

                    df_trip = markdown_to_df(agent_result)
                    if df_trip is not None:
                        trip_file = os.path.join(
                            EXTRACTS_DIR,
                            f"trip_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        )
                        df_trip.to_excel(trip_file, index=False)

                        image_path = df_to_trip_png(df_trip)
                        if image_path:
                            self.send_image(sender, image_path, "📊 Trip Summary")
                        else:
                            self.send_message(
                                sender, f"📊 Trip Summary:\n{agent_result}"
                            )
                    else:
                        self.send_message(sender, f"📊 Trip Summary:\n{agent_result}")

                except Exception as e:
                    logger.error(f"Error during trip analysis: {str(e)}")
                    self.send_message(
                        sender, "❌ Error analyzing trip data. Please try again."
                    )

            elif router_output == "DELETE_AGENT":
                logger.info("Using DELETE_AGENT")
                try:
                    response = create_excel_agent(f"Extract just the Invoice_Number from this request: '{message_text}'. "
                        "Respond ONLY with the number or 'Not found'.")
                    
                    if response.lower() == "not found":
                        self.send_message(
                            sender, "❌ Could not identify invoice number to delete."
                        )
                        return

                    # Create new version without the deleted invoice
                    new_df = df[df["Invoice_Number"] != response]
                    if len(new_df) < len(df):
                        new_file = os.path.join(
                            EXTRACTS_DIR,"extractedData.xlsx",
                        )
                        new_df.to_excel(new_file, index=False)

                        # Update embeddings
                        embed_and_upload_invoice_data()
                        self.send_message(
                            sender, f"✅ Invoice {response} deleted successfully."
                        )
                    else:
                        self.send_message(
                            sender, f"❌ Invoice {response} not found."
                        )

                except Exception as e:
                    logger.error(f"Error during deletion: {str(e)}")
                    self.send_message(
                        sender, "❌ Error deleting invoice. Please try again."
                    )

            elif router_output == "BASE_AGENT":
                logger.info("Using greeting agent")
                response = self._generate_llm_response(full_prompt)
                self.send_message(sender, response)

            else:
                response = "I'm an invoice processing assistant. How can I help with your invoices?"
                self.send_message(sender, response)

            # Store the outgoing message event
            response_event = {
                "sender": self.phone_number,
                "recipient": sender,
                "message": response,
                "timestamp": int(time.time()),
                "type": "outgoing",
            }
            response_key = f"event:{int(time.time())}:{self.phone_number}"
            self.event_db.set(response_key, json.dumps(response_event))

            # Cache the response
            if not has_attachment and not response.startswith("❌"):
                cache_data = {
                    "query": message_text,
                    "response": response,
                    "timestamp": timestamp,
                }
                self.cache_db.set(
                    f"cache:{sender}:{message_text}", json.dumps(cache_data)
                )

            return response

        except Exception as e:
            logger.error(f"Error in process_incoming_message: {str(e)}")
            error_msg = "Sorry, I encountered an error processing your request."
            self.send_message(sender, error_msg)
            return error_msg

    def _generate_llm_response(self, prompt: str) -> str:
        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            raise

    def _get_conversation_history(self, sender: str, limit: int = 5) -> str:
        pattern = f"event:*:{sender}"
        keys = self.event_db.keys(pattern)
        pattern_responses = f"event:*:{self.phone_number}"
        response_keys = self.event_db.keys(pattern_responses)

        all_keys = keys + response_keys
        events = []

        for key in all_keys:
            event_data = json.loads(self.event_db.get(key))
            if (
                event_data.get("sender") == sender
                and event_data.get("type") == "incoming"
            ) or (
                event_data.get("sender") == self.phone_number
                and event_data.get("recipient") == sender
            ):
                events.append(event_data)

        events.sort(key=lambda x: x.get("timestamp", 0))
        recent_events = events[-limit:] if len(events) > limit else events

        history = []
        for event in recent_events:
            prefix = "User" if event.get("type") == "incoming" else "Bot"
            history.append(f"{prefix}: {event.get('message', '')}")

        return "\n".join(history)

    def send_message(self, recipient: str, message: str) -> bool:
        try:
            logger.info(f"Sending message to {recipient}: {message[:30]}...")
            cmd = [
                SIGNAL_CLI_PATH,
                "-u",
                self.phone_number,
                "send",
                "-m",
                message,
                recipient,
            ]
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Failed to send message: {result.stderr}")
                return False

            logger.info(f"Message sent successfully to {recipient}")
            return True
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            logger.exception("Full exception details:")
            return False

    def send_image(self, recipient, image_path, caption=""):
        try:
            recipient = recipient.strip()
            caption = caption.strip().replace("\r", "").replace("\n", " ")
            print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
            print("Recipient:", repr(recipient))
            print("Image path:", repr(image_path))
            print("Caption:", repr(caption))

            command = [
                SIGNAL_CLI_PATH,
                "-u",
                self.phone_number,
                "send",
                "-a",
                image_path,  # 🟢 MOVE -a BEFORE -m
                "-m",
                caption,
                recipient,
            ]
            print("🚀 Subprocess command:", command)

            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"✅ Image sent successfully to {recipient}")
                return True
            else:
                logger.error(f"❌ Failed to send image to {recipient}: {result.stderr}")
                return False
        except Exception as e:
            logger.exception(f"Exception during send_image: {str(e)}")
            return False


def setup_signal_cli(phone_number: str) -> bool:
    try:
        # Check if already registered
        cmd = [SIGNAL_CLI_PATH, "-u", phone_number, "listDevices"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Signal-cli already registered")
            return True

        # Register
        logger.info(f"Registering signal-cli with {phone_number}")
        cmd = [SIGNAL_CLI_PATH, "-u", phone_number, "register"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if "verification required" in result.stdout or "captcha" in result.stdout:
            logger.info(
                "Verification required. Please check your phone for verification code."
            )
            verification_code = input("Enter verification code: ")

            # Verify
            cmd = [SIGNAL_CLI_PATH, "-u", phone_number, "verify", verification_code]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"Verification failed: {result.stderr}")
                return False

            logger.info("Signal-cli verification successful")
            return True
        elif result.returncode != 0:
            logger.error(f"Registration failed: {result.stderr}")
            return False

        logger.info("Signal-cli registration successful")
        return True
    except Exception as e:
        logger.error(f"Error setting up signal-cli: {str(e)}")
        return False


def listen_for_messages(bot: SignalBot):
    try:
        logger.info("Starting message listener...")

        while True:
            try:
                receive_cmd = [
                    SIGNAL_CLI_PATH,
                    "-u",
                    bot.phone_number,
                    "receive",
                    "-t",
                    "5",
                ]
                logger.info(f"Running receive command: {' '.join(receive_cmd)}")
                result = subprocess.run(receive_cmd, capture_output=True, text=True)

                if result.stdout:
                    logger.info(f"Received raw output: {result.stdout}")
                    lines = result.stdout.strip().split("\n")

                    current_sender = None
                    current_timestamp = None
                    current_body = None
                    current_attachments = []

                    for line in lines:
                        if "Envelope from:" in line:
                            match = re.search(r"\+\d+", line)
                            if match:
                                current_sender = match.group(0)

                        elif "Timestamp:" in line and current_sender:
                            match = re.search(r"Timestamp: (\d+)", line)
                            if match:
                                current_timestamp = int(match.group(1))

                        elif "Body:" in line and current_sender and current_timestamp:
                            current_body = line.split("Body:")[1].strip()

                        elif (
                            "Stored plaintext in:" in line
                            and current_sender
                            and current_timestamp
                        ):
                            attachment_path = line.split("Stored plaintext in:")[
                                1
                            ].strip()
                            if os.path.exists(attachment_path):
                                current_attachments.append(attachment_path)
                            else:
                                logger.warning(
                                    f"Attachment path does not exist: {attachment_path}"
                                )

                        elif line.strip() == "" and (
                            current_body or current_attachments
                        ):
                            try:
                                logger.info(f"Processing message from {current_sender}")
                                response = bot.process_incoming_message(
                                    sender=current_sender,
                                    message_text=current_body or "",
                                    timestamp=current_timestamp,
                                    attachments=(
                                        current_attachments
                                        if current_attachments
                                        else None
                                    ),
                                )
                                logger.info(f"Response generated: {response}")
                            except Exception as msg_error:
                                logger.error(
                                    f"Error processing message: {str(msg_error)}"
                                )

                            # Reset
                            current_sender = None
                            current_timestamp = None
                            current_body = None
                            current_attachments = []

                    # Final message check
                    if current_sender and (current_body or current_attachments):
                        try:
                            logger.info(f"Final message block from {current_sender}")
                            response = bot.process_incoming_message(
                                sender=current_sender,
                                message_text=current_body or "",
                                timestamp=current_timestamp,
                                attachments=(
                                    current_attachments if current_attachments else None
                                ),
                            )
                            logger.info(f"Response generated: {response}")
                        except Exception as msg_error:
                            logger.error(
                                f"Error processing final message: {str(msg_error)}"
                            )

                if result.stderr and "Config file is in use" not in result.stderr:
                    logger.warning(f"Stderr from receive command: {result.stderr}")

                time.sleep(1)

            except Exception as loop_error:
                logger.error(f"Error in message polling loop: {str(loop_error)}")
                time.sleep(5)

    except Exception as e:
        logger.error(f"Error in listen_for_messages: {str(e)}")


def main():
    print("Starting Invoice Processing Bot with Local Storage")
    global EMBEDDING_MODEL_NAME

    # Initialize local storage
    if not os.path.exists(LOCAL_STORAGE_ROOT):
        os.makedirs(LOCAL_STORAGE_ROOT)

    parser = argparse.ArgumentParser(
        description="Signal Invoice Bot with Local Storage"
    )
    parser.add_argument(
        "--phone",
        default=PHONE_NUMBER,
        help="Phone number for Signal (with country code)",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh invoice data from local storage",
    )

    args = parser.parse_args()

    # Refresh invoice data if requested
    if args.refresh_data:
        try:
            embed_and_upload_invoice_data()
        except Exception as e:
            logger.error(f"Error refreshing data: {str(e)}")

    if not setup_signal_cli(args.phone):
        logger.error("Failed to set up signal-cli. Exiting.")
        return

    # Initialize the bot
    bot = SignalBot(args.phone)

    # Send test message if target number is configured
    # test_number = TARGET_PHONE_NUMBER
    # if test_number:
    #   logger.info(f"Testing message send to {test_number}")
    #  result = bot.send_message(
    #     test_number,
    #    "Invoice Bot is starting up with local storage and ready to process your invoice queries!",
    # )
    # logger.info(f"Test message send result: {result}")

    # Start listening for messages
    listen_for_messages(bot)


if __name__ == "__main__":
    main()
