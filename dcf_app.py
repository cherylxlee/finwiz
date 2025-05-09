import streamlit as st
import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.llms import Ollama
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI

from dcf_analyze import dcf_calculator_ui

# ------------------------------
# Core Configuration Module
# ------------------------------

load_dotenv()

def initialize_app_config():
    """Initialize application configuration from environment variables."""
    config = {
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT"),
        "index_name": os.getenv("INDEX_NAME"),
        "cohere_api_key": os.getenv("COHERE_API_KEY"),
        "azure_api_key": os.getenv("AZURE_API_KEY")
    }
    
    # Validate required configuration
    missing_configs = [k for k, v in config.items() if not v]
    if missing_configs:
        st.warning(f"Missing configuration: {', '.join(missing_configs)}")
        
    return config

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(page_title="FinWiz: Financial Data Extraction", layout="wide")
    st.title("FinWiz: Financial Data Extraction")

def get_company_configs():
    """Get company configurations and mappings."""
    # All available companies
    all_companies = ["Alphabet", "Amazon", "Apple", "Meta", "Microsoft", "Nvidia", "Tesla"]
    
    # Companies supported for DCF analysis (excluding Amazon and Nvidia)
    dcf_supported_companies = ["Alphabet", "Apple", "Meta", "Microsoft", "Tesla"]
    
    # Namespace mappings for vectorstore
    namespaces = {
        "Apple": "apple_10k",
        "Nvidia": "nvidia_10k",
        "Meta": "meta_10k",
        "Microsoft": "microsoft_10k",
        "Tesla": "tesla_10k",
        "Alphabet": "google_10k",
        "Amazon": "amazon_10k"
    }
    
    # Map fiscal years - 2025 for Nvidia, 2024 for others
    def get_fiscal_year(company):
        return 2025 if company.lower() == "nvidia" else 2024
    
    return {
        "all_companies": all_companies,
        "dcf_supported_companies": dcf_supported_companies,
        "namespaces": namespaces,
        "get_fiscal_year": get_fiscal_year
    }

# ------------------------------
# Model Integration Module
# ------------------------------

def get_llm(model_name="qwen3:14b"):
    """Initialize the Ollama LLM."""
    return Ollama(model=model_name, base_url="http://localhost:11434")

def get_azure_gpt4o_client():
    """Initialize Azure OpenAI client with GPT-4o model."""
    AZURE_OPENAI_VERSION = "2024-05-01-preview"
    client = AzureOpenAI(
        base_url=f"https://dsa-usf-api25.azure-api.net/chat/completions?api-version={AZURE_OPENAI_VERSION}",
        default_headers={"Ocp-Apim-Subscription-Key": os.getenv("AZURE_API_KEY")},
        api_key="placeholder",
        api_version=AZURE_OPENAI_VERSION,
    )
    return client

def filter_thinking(response):
    """Remove <think>...</think> sections from LLM responses."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

# ------------------------------
# Vector Store Module
# ------------------------------

def initialize_vector_store(config):
    """Initialize the Pinecone vector store."""
    if not all([config["pinecone_api_key"], config["cohere_api_key"], 
               config["index_name"], config["pinecone_environment"]]):
        st.warning("Please make sure all environment variables are set.")
        return None, None
    
    try:
        # Setup embedding function with updated model
        embeddings = CohereEmbeddings(
            model="embed-v4.0",
            cohere_api_key=config["cohere_api_key"]
        )
        
        # Initialize Pinecone client
        pc = Pinecone(
            api_key=config["pinecone_api_key"],
            environment=config["pinecone_environment"]
        )
        
        # List indexes to verify the one we need exists
        existing = pc.list_indexes().names()
        
        # Create index if it doesn't exist with updated dimension
        if config["index_name"] not in existing:
            st.warning(f"Index {config['index_name']} not found. Creating it...")
            pc.create_index(
                name=config["index_name"],
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=config["pinecone_environment"]
                )
            )
        
        index = pc.Index(config["index_name"])
        
        return index, pc, embeddings
        
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return None, None, None

def get_vector_store(index, embeddings, namespace):
    """Get configured vector store with the specified namespace."""
    try:
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="chunk_content",
            namespace=namespace
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# ------------------------------
# Document Processing Module
# ------------------------------

def detect_table_in_text(content):
    """Detect if text content likely contains a table structure."""
    # Check for common table indicators
    table_indicators = [
        # Pipe-based tables - most common indicator
        content.count("|") > 5 and content.count("\n") > 3,
        
        # Contains rows with multiple aligned numbers
        bool(re.search(r"\$\s*[\d,]+\s*\|\s*\$\s*[\d,]+", content)),
        
        # Contains headers like "2024 | 2023 | 2022" or year columns
        bool(re.search(r"202[0-5]\s*[|\t].*?202[0-5]", content)),
        
        # Common financial table keywords with numbers
        "(in millions)" in content.lower() and bool(re.search(r"\d{3,}", content)),
        
        # Net sales/revenue tables
        "net sales" in content.lower() and "total" in content.lower() and bool(re.search(r"\d{5,}", content)),
        
        # Operating income specific detection
        "operating income" in content.lower() and bool(re.search(r"\$\s*[\d,]+", content)),
        
        # Consolidated statement indicators
        "consolidated statement" in content.lower() and bool(re.search(r"\d{4,}", content))
    ]
    
    # Return True if any indicator is found
    return any(table_indicators)

def format_table_content(content):
    """Format table content for better display."""
    # Check if content looks like a pipe table
    if "|" in content and content.count("\n") > 2:
        # Try to preserve pipe table format for markdown
        lines = content.split("\n")
        formatted_lines = []
        
        for line in lines:
            if "|" in line:
                formatted_lines.append(line)
            else:
                # Add context lines that aren't part of the table
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    # If not a pipe table, just return as is
    return content

def process_search_results(docs):
    """Process search results to handle different content types, with table detection."""
    results = []
    for i, doc in enumerate(docs):
        content_type = doc.metadata.get("type", "text")
        
        # Check for tables in text content
        has_table = False
        if content_type == "text":
            has_table = detect_table_in_text(doc.page_content)
        
        if content_type == "text":
            # Handle text content
            pages = doc.metadata.get("pages", [])
            page_display = f"Pages: {', '.join(str(p) for p in pages)}" if pages else "Page: unknown"
            
            results.append({
                "type": "text",
                "content": doc.page_content,
                "page_info": page_display,
                "has_table": has_table,
                "metadata": doc.metadata
            })
        elif content_type == "image":
            # Handle image content
            page = doc.metadata.get("pages", "unknown")
            image_url = doc.page_content  # Assuming this contains the GCP URL
            results.append({
                "type": "image",
                "content": image_url,
                "page_info": f"Page: {page}",
                "has_table": False,
                "metadata": doc.metadata
            })
    
    results.sort(key=lambda x: x.get("has_table", False), reverse=True)
    
    return results

def is_financial_table(content, metric):
    """Check if content likely contains financial table data for the specific metric."""
    # Implement more sophisticated detection logic
    has_numbers = sum(c.isdigit() for c in content) > 20
    has_years = any(year in content for year in ['2021', '2022', '2023', '2024', '2025'])
    
    # Metric-specific patterns
    metric_patterns = {
        "Revenue": ["revenue", "net sales", "total revenue"],
        "Operating_Income": ["operating income", "income from operations"],
        "Tax_Rate": ["tax rate", "effective tax", "income tax"],
        "CapEx": ["capital expenditure", "property and equipment", "ppe"],
        "Depreciation": ["depreciation", "amortization"]
    }
    
    pattern_match = any(pattern.lower() in content.lower() for pattern in metric_patterns.get(metric, []))
    
    return has_numbers and has_years and pattern_match

# ------------------------------
# Question Answering Module
# ------------------------------

def create_financial_qa_prompt(company):
    """Create a prompt template for financial data extraction."""
    prompt_template = """
    You are a financial analyst extracting information from 10-K documents for {company}.
    
    CONTEXT INFORMATION:
    {context}
    
    QUESTION:
    {question}
    
    TASK:
    - Use ONLY the information in the context above to answer the question.
    - Be precise with numbers, including units (e.g., $M, $B, %, etc.).
    - If data from multiple years is available, clearly show the trend.
    - If the information includes tables or financial statements, parse them correctly.
    - If you see page numbers in the context, mention them in your response.
    - If the information is not available in the context, say "The information is not available in the provided context."
    
    OUTPUT FORMAT:
    Start with a brief direct answer to the question.
    Then provide supporting details and context.
    Include source pages if available.
    """
    
    return PromptTemplate(
        template=prompt_template.replace("{company}", company),
        input_variables=["context", "question"]
    )

def create_financial_qa_prompt_gpt4o(company, context, question):
    """Create a GPT-4o prompt for financial data extraction."""
    return f"""
You are a financial analyst extracting information from 10-K documents for {company}.
    
CONTEXT INFORMATION:
{context}
    
QUESTION:
{question}
    
TASK:
- Use ONLY the information in the context above to answer the question.
- Be precise with numbers, including units (e.g., $M, $B, %, etc.).
- If data from multiple years is available, clearly show the trend.
- If the information includes tables or financial statements, parse them correctly.
- If you see page numbers in the context, mention them in your response.
- If the information is not available in the context, say "The information is not available in the provided context."
    
OUTPUT FORMAT:
Start with a brief direct answer to the question.
Then provide supporting details and context.
Include source pages if available.
"""

def execute_financial_query(llm, vectorstore, prompt, user_query, k_value=5):
    """Execute a financial query using a retrieval QA chain."""
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": k_value}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        result = qa.invoke({"query": user_query})
        
        if '<think>' in result['result']:
            result['result'] = filter_thinking(result['result'])
        
        return result
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return None

def execute_financial_query_with_gpt4o(client, vectorstore, company, user_query, k_value=5):
    """Execute a financial query using GPT-4o with RAG."""
    try:
        docs = vectorstore.similarity_search(user_query, k=k_value)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = create_financial_qa_prompt_gpt4o(company, context, user_query)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst expert in extracting and interpreting data from 10-K filings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        result = {
            "result": answer,
            "source_documents": docs
        }
        
        return result
    except Exception as e:
        st.error(f"Error executing GPT-4o query: {str(e)}")
        return None

# ------------------------------
# DCF Analysis Module
# ------------------------------

def company_specific_retrieval(vectorstore, company, fiscal_year):
    """
    Company-specific retrieval approach that targets the unique formatting
    and terminology of each supported company.
    """
    # Default to the general approach
    results = improved_financial_metrics_retrieval(vectorstore, company, fiscal_year)
    
    # Apply company-specific enhancements
    if company.lower() == "microsoft":
        microsoft_results = microsoft_specific_retrieval(vectorstore, fiscal_year)
        results = {**results, **microsoft_results}
    elif company.lower() == "alphabet":
        alphabet_results = alphabet_specific_retrieval(vectorstore, fiscal_year)
        results = {**results, **alphabet_results}
    elif company.lower() == "tesla":
        tesla_results = tesla_specific_retrieval(vectorstore, fiscal_year)
        results = {**results, **tesla_results}
    
    return results

def microsoft_specific_retrieval(vectorstore, fiscal_year):
    """
    Microsoft-specific retrieval for problematic metrics (CapEx and Depreciation).
    """
    results = {}
    
    # Special Microsoft CapEx queries
    capex_queries = [
        "Additions to property and equipment",
        "microsoft cash flow statement investing activities",
        f"microsoft investing activities {fiscal_year}",
        "property and equipment microsoft",
        "capital expenditures microsoft"
    ]
    
    # Special Microsoft Depreciation queries
    depreciation_queries = [
        "Depreciation, amortization, and other",
        "Depreciation, amortization, and other | 22,287 | 13,861 | 14,460 "
    ]
    
    # Execute CapEx queries
    capex_docs = []
    for query in capex_queries:
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        # Filter for investing section
        filtered_docs = [doc for doc in retrieved_docs if 
                        any(term in doc.page_content.lower() for term in 
                           ["investing", "additions to property", "capital expenditure"])]
        capex_docs.extend(filtered_docs)
    
    # Execute Depreciation queries
    depreciation_docs = []
    for query in depreciation_queries:
        retrieved_docs = vectorstore.similarity_search(query, k=10)
        # Filter for depreciation mentions
        filtered_docs = [doc for doc in retrieved_docs if 
                        any(term in doc.page_content.lower() for term in 
                           ["amortization", "cash flows"])]
        depreciation_docs.extend(filtered_docs)
    
    # Handle potential empty results
    if capex_docs:
        unique_capex_contents = set(doc.page_content for doc in capex_docs)
        results["CapEx"] = "\n\n".join(unique_capex_contents)
    
    if depreciation_docs:
        unique_depreciation_contents = set(doc.page_content for doc in depreciation_docs)
        results["Depreciation"] = "\n\n".join(unique_depreciation_contents)
    
    return results

def alphabet_specific_retrieval(vectorstore, fiscal_year):
    """
    Alphabet (Google) specific retrieval for problematic metrics
    (Revenue, Operating Income, and Tax Rate).
    """
    results = {}
    
    # Alphabet uses specific table structures
    income_statement_queries = [
        "alphabet consolidated statements of income",
        "alphabet revenues",
        f"alphabet income from operations {fiscal_year}",
        "google consolidated income statement"
    ]
    
    tax_rate_queries = [
        "alphabet effective tax rate",
        "alphabet reconciliation of federal statutory income tax rate",
        "google effective income tax rate",
        f"alphabet income taxes {fiscal_year}"
    ]
    
    # Execute income statement queries to get both revenue and operating income
    income_docs = []
    for query in income_statement_queries:
        retrieved_docs = vectorstore.similarity_search(query, k=5)
        # Filter for likely income statement tables
        filtered_docs = [doc for doc in retrieved_docs if 
                        is_likely_income_statement(doc.page_content)]
        income_docs.extend(filtered_docs)
    
    # Execute tax rate queries
    tax_docs = []
    for query in tax_rate_queries:
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        # Filter for likely tax rate tables
        filtered_docs = [doc for doc in retrieved_docs if 
                        is_likely_tax_table(doc.page_content)]
        tax_docs.extend(filtered_docs)
    
    # Extract and assign the metrics
    revenue_docs = []
    operating_income_docs = []
    
    for doc in income_docs:
        content = doc.page_content.lower()
        # Check if it contains revenue information
        if "revenues" in content and contains_financial_table_row(content, "revenues"):
            revenue_docs.append(doc)
        
        # Check if it contains operating income information
        if "income from operations" in content and contains_financial_table_row(content, "income from operations"):
            operating_income_docs.append(doc)
    
    # Handle potential empty results
    if revenue_docs:
        unique_revenue_contents = set(doc.page_content for doc in revenue_docs)
        results["Revenue"] = "\n\n".join(unique_revenue_contents)
    
    if operating_income_docs:
        unique_operating_contents = set(doc.page_content for doc in operating_income_docs)
        results["Operating_Income"] = "\n\n".join(unique_operating_contents)
    
    if tax_docs:
        unique_tax_contents = set(doc.page_content for doc in tax_docs)
        results["Tax_Rate"] = "\n\n".join(unique_tax_contents)
    
    return results

def tesla_specific_retrieval(vectorstore, fiscal_year):
    results = {}
    
    tesla_queries = [
        "Income from operations",
        f"tesla Total operating expenses Income from operations {fiscal_year}"
    ]
    
    operating_docs = []
    for query in tesla_queries:
        retrieved_docs = vectorstore.similarity_search(query, k=10)
        filtered_docs = [doc for doc in retrieved_docs if 
                        any(term in doc.page_content.lower() for term in 
                           ["operations", "interest income"])]
        operating_docs.extend(filtered_docs)
        
    if operating_docs:
        unique_capex_contents = set(doc.page_content for doc in operating_docs)
        results["Operating_Income"] = "\n\n".join(unique_capex_contents)
        
    return results

def is_likely_income_statement(content):
    """Check if the content is likely to be from an income statement."""
    content_lower = content.lower()
    
    # Check for income statement indicators
    has_income_statement_header = any(header in content_lower for header in 
                                    ["consolidated statements of income", 
                                     "income statement", 
                                     "statements of operations"])
    
    # Check for typical income statement line items
    has_income_items = sum(1 for item in ["revenues", "cost of", "operating", "net income", 
                                          "income from operations", "expenses"] 
                           if item in content_lower) >= 3
    
    # Check for table structure
    has_table_structure = "|" in content and content.count("|") > 10
    
    return (has_income_statement_header or has_income_items) and has_table_structure

def is_likely_tax_table(content):
    """Check if the content is likely to be from a tax rate reconciliation table."""
    content_lower = content.lower()
    
    # Check for tax table indicators
    has_tax_header = any(header in content_lower for header in 
                        ["effective tax rate", 
                         "reconciliation of", 
                         "income tax rate",
                         "statutory tax"])
    
    # Check for typical tax reconciliation items
    has_tax_items = sum(1 for item in ["federal", "statutory", "effective", 
                                       "foreign", "state", "tax rate"] 
                        if item in content_lower) >= 3
    
    # Check for percentage signs or typical tax rate patterns
    has_percentage_signs = "%" in content
    
    return has_tax_header and (has_tax_items or has_percentage_signs)

def contains_financial_table_row(content, row_name):
    """
    Check if content contains a financial table row with the specified name
    and numeric values (likely financial figures).
    """
    # Normalize row name for case-insensitive search
    row_name_lower = row_name.lower()
    
    # Get lines from content
    lines = content.lower().split('\n')
    
    for line in lines:
        if row_name_lower in line:
            # Check if the line contains numeric values
            has_numbers = bool(re.search(r'\d[\d,\.]+', line))
            if has_numbers:
                return True
    
    return False

def retrieve_financial_metrics(vectorstore, company, fiscal_year):
    """Retrieve contexts for each financial metric separately."""
    
    section_queries = {
        "Revenue": [
            f"revenues {fiscal_year}",
            f"total net sales {fiscal_year}",
            f"net revenue {fiscal_year}",
            f"total revenues {fiscal_year}"
        ],
        "Operating_Income": [
            f"operating income {fiscal_year}",
            "Income from operations |",
            f"operating profit {fiscal_year}"
        ],
        "Tax_Rate": [
            f"effective tax rate {fiscal_year}",
            f"effective rate {fiscal_year}"
        ],
        "CapEx": [
            "Additions to property and equipment |",
            f"{company} cash flow statement payments for acquisition of property plant equipment {fiscal_year}",
            f"{company} cash flow statement capital expenditures {fiscal_year}",
            f"{company} investing activities property equipment {fiscal_year}",
            f"{company} purchases of property and equipment {fiscal_year}",
            f"{company} Purchases related to property and equipment and intangible assets {fiscal_year}"
        ],
        "Depreciation": [
            "Cash flows from operating activities depreciation and amortization",
            "Depreciation, amortization, and other |",
            f"{company} cash flow statement depreciation and amortization {fiscal_year}",
            f"{company} operating activities depreciation {fiscal_year}",
            f"{company} depreciation, amortization, and other",
            f"{company} consolidated statements of cash flows depreciation {fiscal_year}"
        ]
    }
    
    metric_contexts = {}
    
    for metric, queries in section_queries.items():
        for query in queries:
            # Increase k for cash flow metrics to improve recall
            k_value = 5 if metric in ["CapEx", "Depreciation"] else 3
            
            try:
                retrieved_docs = vectorstore.similarity_search(query, k=k_value)
                
                # Apply more specific filtering based on the pattern you found
                if metric == "CapEx":
                    filtered_docs = [doc for doc in retrieved_docs if any(term in doc.page_content.lower() for term in 
                                    ["acquisition of property", "capital expenditure", "purchases of property", 
                                     "investing activities", "property, plant and equipment"])]
                elif metric == "Depreciation":
                    filtered_docs = [doc for doc in retrieved_docs if any(term in doc.page_content.lower() for term in 
                                    ["depreciation and amortization", "depreciation of property", 
                                     "operating activities", "cash generated by operating activities"])]
                else:
                    filtered_docs = [doc for doc in retrieved_docs if is_financial_table(doc.page_content, metric)]
                
                if filtered_docs:
                    if metric not in metric_contexts:
                        metric_contexts[metric] = []
                    metric_contexts[metric].extend(filtered_docs)
            except Exception as e:
                st.warning(f"Error retrieving {metric} data: {str(e)}")
    
    # Convert to text and deduplicate
    result = {}
    for metric, docs in metric_contexts.items():
        # Get unique document contents
        unique_contents = set(doc.page_content for doc in docs)
        result[metric] = "\n\n".join(unique_contents)
    
    return result

def extract_cashflow_metrics(vectorstore, company):
    """Specifically target cash flow statement chunks."""
    
    # First, find the cash flow statement
    cashflow_queries = [
        f"{company} consolidated statements of cash flows",
        f"{company} cash flow statement",
        f"{company} cash generated by operating activities"
    ]
    
    cashflow_docs = []
    for query in cashflow_queries:
        try:
            retrieved_docs = vectorstore.similarity_search(query, k=10)
            cashflow_docs.extend(retrieved_docs)
        except Exception as e:
            st.warning(f"Error retrieving cash flow data: {str(e)}")
    
    # Extract metrics from cash flow docs
    capex_chunks = []
    depreciation_chunks = []
    
    for doc in cashflow_docs:
        content = doc.page_content.lower()
        
        # Check for CapEx patterns
        if any(pattern in content for pattern in [
            "acquisition of property", "capital expenditure", 
            "purchases of property", "property, plant and equipment",
            "investing activities"
        ]):
            capex_chunks.append(doc.page_content)
        
        # Check for Depreciation patterns
        if any(pattern in content for pattern in [
            "depreciation and amortization", 
            "adjustments to reconcile", 
            "operating activities"
        ]):
            depreciation_chunks.append(doc.page_content)
    
    return {
        "CapEx": "\n\n".join(capex_chunks),
        "Depreciation": "\n\n".join(depreciation_chunks)
    }

def direct_pattern_search(vectorstore, company):
    """Direct pattern search for cash flow table rows."""
    
    # Specialized queries for the exact patterns in Apple's 10-K
    specialized_queries = [
        # Cash flow statement investing section
        f"{company} investing activities payments for acquisition of property",
        # Cash flow statement operating section
        f"{company} operating activities depreciation and amortization",
        # Generic cash flow table markers
        f"{company} cash, cash equivalents beginning balances"
    ]
    
    all_retrieved_docs = []
    for query in specialized_queries:
        try:
            docs = vectorstore.similarity_search(query, k=5)
            all_retrieved_docs.extend(docs)
        except Exception as e:
            st.warning(f"Error in pattern search: {str(e)}")
    
    # Now analyze the retrieved text for table patterns
    capex_matches = []
    depreciation_matches = []
    
    for doc in all_retrieved_docs:
        content = doc.page_content
        
        # Look for table rows with CapEx terms
        if re.search(r'(payments for acquisition of property|capital expenditures|purchases of property).+?(\d[\d,\.]+)', content, re.IGNORECASE):
            capex_matches.append(content)
        
        # Look for table rows with Depreciation terms
        if re.search(r'(depreciation and amortization|depreciation of property).+?(\d[\d,\.]+)', content, re.IGNORECASE):
            depreciation_matches.append(content)
    
    return {
        "CapEx_Pattern_Matches": "\n\n".join(capex_matches),
        "Depreciation_Pattern_Matches": "\n\n".join(depreciation_matches)
    }

def improved_financial_metrics_retrieval(vectorstore, company, fiscal_year):
    """Improved retrieval pipeline for financial metrics."""
    
    # Standard retrieval for all metrics
    general_results = retrieve_financial_metrics(vectorstore, company, fiscal_year)
    
    # Special handling for cash flow metrics
    cashflow_results = extract_cashflow_metrics(vectorstore, company)
    pattern_results = direct_pattern_search(vectorstore, company)
    
    # Combine results
    combined_results = {}
    
    for metric in ["Revenue", "Operating_Income", "Tax_Rate"]:
        combined_results[metric] = general_results.get(metric, "No relevant information found.")
    
    # For CapEx, combine all approaches
    capex_texts = [
        general_results.get("CapEx", ""),
        cashflow_results.get("CapEx", ""),
        pattern_results.get("CapEx_Pattern_Matches", "")
    ]
    combined_results["CapEx"] = "\n\n".join([t for t in capex_texts if t])
    
    # For Depreciation, combine all approaches
    dep_texts = [
        general_results.get("Depreciation", ""),
        cashflow_results.get("Depreciation", ""),
        pattern_results.get("Depreciation_Pattern_Matches", "")
    ]
    combined_results["Depreciation"] = "\n\n".join([t for t in dep_texts if t])
    
    # Clean up empty results
    for metric, content in combined_results.items():
        if not content:
            combined_results[metric] = "No relevant information found."
    
    return combined_results

def preprocess_context(context, metric_name):
    """Preprocess context to make it more readable for the LLM."""
    # Fix markdown tables to make them more readable
    processed = context.replace('|', ' | ')
    processed = re.sub(r'\s+\|\s+', ' | ', processed)
    
    # Handle dollar signs with space
    processed = re.sub(r'\$\s+', '$', processed)
    processed = re.sub(r'\$(\d)', '$ \\1', processed)
    
    # Make percentages more explicit, especially for tax rates
    if metric_name == "Tax_Rate":
        # Find percentages and make them clearer
        processed = re.sub(r'(\d+)\.(\d+)%', r'\1.\2 percent', processed)
        processed = re.sub(r'(\d+)%', r'\1 percent', processed)
    
    # Clean up extra spaces and newlines
    processed = re.sub(r'\s+', ' ', processed)
    processed = re.sub(r'\n{3,}', '\n\n', processed)
    
    return processed

def create_extraction_prompt(company, metric_contexts, fiscal_year):
    """Create a standard extraction prompt for most companies."""
    years = [str(fiscal_year), str(fiscal_year-1), str(fiscal_year-2)]
    
    # Process contexts
    processed_contexts = {}
    for metric, context in metric_contexts.items():
        processed_contexts[metric] = preprocess_context(context, metric)
    
    # Create metric prompts
    metric_prompts = []
    for metric, context in processed_contexts.items():
        metric_prompts.append(f"CONTEXT FOR {metric.upper()}:\n{context}\n")
    
    all_contexts = "\n".join(metric_prompts)
    
    return f"""
You are FinancialGPT, a financial data extraction expert.

TASK:
Extract {company}'s financial metrics from the financial data provided. Each metric has its own context section.

METRICS TO EXTRACT:
1. Revenue (look for "Revenues", "Total net sales", "Net revenue", "Total revenues", etc.)
2. Operating income (look for "Operating income", "Income from operations", "Total operating income", etc.)
3. Capital expenditures (look for "Payments for acquisition of property, plant and equipment", "Purchases of property and equipment", "Additions to property and equiment", etc.)
4. Depreciation and amortization (look for "Depreciation and amortization", "Depreciation of property and equipment", "Depreciation, amortization and impairment", etc.)
5. Effective tax rate (look for "Effective tax rate" - extract the percentage value, e.g. extract "24.1%" as "24.1")

DATA EXTRACTION RULES:
- Extract plain numbers WITHOUT commas, currency symbols, or punctuation
- For financial values, ensure they are in millions (convert if necessary)
- For tax rates, extract the percentage value (e.g., extract "24.1%" as "24.1")
- When extracting from tables, carefully match the year to the correct value
- If you can't find a value, use "N/A"

SPECIAL INSTRUCTIONS:
- Pay close attention to tables - they contain most of the key data
- Carefully verify which year each value corresponds to
- For tax rates, look for the "Effective tax rate" line rather than the statutory rate
- Be precise with units (millions vs billions)

FINANCIAL DATA:
{all_contexts}

OUTPUT FORMAT (JSON):
{{
    "Units": "In millions (or whatever unit is specified)",
    "Revenue": {{
        "{years[0]}": value,
        "{years[1]}": value,
        "{years[2]}": value
    }},
    "Operating_Income": {{
        "{years[0]}": value,
        "{years[1]}": value,
        "{years[2]}": value
    }},
    "CapEx": {{
        "{years[0]}": value,
        "{years[1]}": value,
        "{years[2]}": value
    }},
    "Depreciation": {{
        "{years[0]}": value,
        "{years[1]}": value,
        "{years[2]}": value
    }},
    "Tax_Rate": {{
        "{years[0]}": value,
        "{years[1]}": value,
        "{years[2]}": value
    }}
}}

Output ONLY the JSON, no explanations.
"""

# ------------------------------
# Data Presentation Module
# ------------------------------

def display_results(results):
    """Display search results accounting for different content types."""
    for i, item in enumerate(results):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Add table indicator to title if present
            title = f"Document {i+1} - {item['page_info']}"
            if item.get("has_table", False):
                title += " [CONTAINS TABLE]"
                
            with st.expander(title, expanded=i==0):
                if item["type"] == "text":
                    if item.get("has_table", False):
                        formatted_content = format_table_content(item["content"])
                        st.markdown(formatted_content)
                    else:
                        st.markdown(item["content"])
                elif item["type"] == "image":
                    st.image(item["content"], caption=f"Image from {item['page_info']}")
        
        with col2:
            st.caption("Metadata")
            for key, value in item["metadata"].items():
                if key not in ["type", "pages"]:
                    st.caption(f"{key}: {value}")

def format_metric_name(name):
    """Format metrics keys for display (e.g., 'Operating_Income' -> 'Operating Income')."""
    return name.replace("_", " ")

def create_metrics_table(dcf_data, years, include_keys=None, exclude_keys=None):
    """Create a formatted table of financial metrics with proper formatting."""
    
    # Filter metrics
    if include_keys:
        metrics = [key for key in include_keys if key in dcf_data]
    elif exclude_keys:
        metrics = [key for key in dcf_data.keys() if key not in exclude_keys and key != "Units"]
    else:
        metrics = [key for key in dcf_data.keys() if key != "Units"]
    
    # Create DataFrame data
    data = []
    
    for metric in metrics:
        row = [format_metric_name(metric)]
        
        # Add values for each year
        for year in years:
            if year in dcf_data[metric]:
                year_data = dcf_data[metric][year]
                
                # Handle both dictionary and direct value formats
                if isinstance(year_data, dict) and "value" in year_data:
                    raw_value = year_data["value"]
                else:
                    raw_value = year_data
                
                # Convert string values to numbers if possible
                if isinstance(raw_value, str):
                    # Try to convert to number, handling commas
                    try:
                        cleaned_value = raw_value.replace(',', '')
                        if cleaned_value.replace('.', '', 1).isdigit():
                            # It's a number
                            if '.' in cleaned_value:
                                value = float(cleaned_value)
                            else:
                                value = int(cleaned_value)
                        else:
                            # Not a number, keep as is
                            value = raw_value
                    except:  # noqa: E722
                        value = raw_value
                else:
                    value = raw_value
                
                # Format based on metric type
                metric_lower = metric.lower().replace('_', ' ')
                if "rate" in metric_lower or "growth" in metric_lower:
                    # Percentage metrics
                    if isinstance(value, (int, float)):
                        row.append(f"{value:.2f}%")
                    elif isinstance(value, str):
                        try:
                            # Remove commas and % symbols
                            cleaned = value.replace(',', '')
                            if "%" in cleaned:
                                # Convert percentage string to decimal
                                numeric_part = cleaned.replace('%', '')
                                value = float(numeric_part)
                            elif cleaned.replace('.', '', 1).isdigit():
                                # Convert to number
                                if '.' in cleaned:
                                    value = float(cleaned)
                                else:
                                    value = int(cleaned)
                            row.append(f"{value:.2f}%")
                        except Exception as e:  # noqa: F841
                            row.append(value)  # Keep as string if conversion fails
                    else:
                        row.append(value)
                elif isinstance(value, (int, float)):
                    # Regular numeric metrics - format with commas
                    row.append(f"{value:,}")
                else:
                    # Non-numeric
                    row.append(value)
            else:
                row.append("N/A")
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["Metric"] + years)
    
    # Display the table
    st.dataframe(df, use_container_width=True)
    
    return df

def prepare_csv_export(dcf_data, years):
    """Prepare DCF data for CSV export with raw values and proper conversions."""
    header = ["Metric"] + years
    
    # Create data rows with raw values
    rows = []
    for metric, values in dcf_data.items():
        if metric != "Units":  # Skip the Units entry in output rows
            row = [metric]
            metric_lower = metric.lower().replace('_', ' ')
            is_percentage = any(term in metric_lower for term in ["rate", "growth", "%"])
            
            for year in years:
                if year in values:
                    # Extract raw value
                    year_data = values[year]
                    if isinstance(year_data, dict) and "value" in year_data:
                        value = year_data["value"]
                    else:
                        value = year_data
                    
                    # Skip N/A values
                    if value == "N/A" or value == "None":
                        row.append("N/A")
                        continue
                    
                    # Convert string values to numeric if possible
                    if isinstance(value, str):
                        try:
                            # Remove commas and % symbols
                            cleaned = value.replace(',', '')
                            if "%" in cleaned:
                                # Convert percentage string to decimal
                                numeric_part = cleaned.replace('%', '')
                                value = float(numeric_part) / 100
                            elif cleaned.replace('.', '', 1).isdigit():
                                # Convert to number
                                if '.' in cleaned:
                                    value = float(cleaned)
                                else:
                                    value = int(cleaned)
                        except Exception:
                            pass  # Keep as string if conversion fails
                    
                    # Apply conversions for numeric values
                    if isinstance(value, (int, float)):
                        if is_percentage:
                            # Convert percentage to decimal
                            value = value / 100
                    
                    row.append(value)
                else:
                    row.append("N/A")
            rows.append(row)
    
    # Create CSV string
    csv_string = ",".join(header) + "\n"
    for row in rows:
        csv_string += ",".join(str(cell) for cell in row) + "\n"
    
    return csv_string

def create_metric_context_explorer(metric_contexts):
    """Display metric contexts in a structured way."""
    for metric, context in metric_contexts.items():
        with st.expander(f"{metric} Context", expanded=False):
            st.text(context)
            st.divider()

# ------------------------------
# LLM Data Extraction Module
# ------------------------------

def process_with_gpt4o(client, company, metric_contexts, fiscal_year):
    """Process financial data extraction with GPT-4o."""
    prompt = create_extraction_prompt(company, metric_contexts, fiscal_year)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are FinancialGPT, a financial data extraction expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Lower temperature for more accurate extraction
        )
        
        # Extract JSON from response
        result_text = response.choices[0].message.content
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, result_text)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                dcf_data = json.loads(json_str)
                return dcf_data
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON: {str(e)}")
                st.code(json_str)
                return None
        else:
            st.error("Could not extract JSON from GPT-4o response")
            st.code(result_text)
            return None
    except Exception as e:
        st.error(f"Error using GPT-4o: {str(e)}")
        return None

def process_with_ollama(llm, company, metric_contexts, fiscal_year):
    """Process financial data extraction with Ollama LLM."""
    prompt = create_extraction_prompt(company, metric_contexts, fiscal_year)
    
    try:
        response = llm.invoke(prompt)
        
        # Extract JSON from response
        json_pattern = r'\{[\s\S]*\}'
        json_match = re.search(json_pattern, response)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                dcf_data = json.loads(json_str)
                return dcf_data
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON: {str(e)}")
                st.code(json_str)
                return None
        else:
            st.error("Could not extract JSON from Ollama response")
            st.code(response)
            return None
    except Exception as e:
        st.error(f"Error using Ollama: {str(e)}")
        return None

def calculate_revenue_growth(dcf_data):
    """Calculate revenue growth rates based on extracted revenue data."""
    if "Revenue" in dcf_data:
        years = sorted(dcf_data["Revenue"].keys(), reverse=True)
        
        # Make sure we have at least 2 years of data to calculate growth
        if len(years) >= 2:
            # Add Revenue Growth to the JSON
            dcf_data["Revenue_Growth"] = {}
            
            for i in range(len(years)-1):
                current_year = years[i]
                previous_year = years[i+1]
                
                if dcf_data["Revenue"][current_year] != "N/A" and dcf_data["Revenue"][previous_year] != "N/A":
                    try:
                        current_value = float(dcf_data["Revenue"][current_year])
                        previous_value = float(dcf_data["Revenue"][previous_year])
                        
                        # Calculate growth percentage
                        if previous_value != 0:
                            growth = ((current_value - previous_value) / previous_value) * 100
                            # Round to 2 decimal places
                            dcf_data["Revenue_Growth"][current_year] = round(growth, 4)
                        else:
                            dcf_data["Revenue_Growth"][current_year] = "N/A"
                    except (ValueError, TypeError):
                        dcf_data["Revenue_Growth"][current_year] = "N/A"
                else:
                    dcf_data["Revenue_Growth"][current_year] = "N/A"
            
            # Add N/A for the earliest year since we can't calculate growth
            dcf_data["Revenue_Growth"][years[-1]] = "N/A"
    
    return dcf_data

# ------------------------------
# Main Application Logic
# ------------------------------

def main():
    setup_page_config()
    
    config = initialize_app_config()
    company_configs = get_company_configs()
    
    vector_store_components = initialize_vector_store(config)
    
    with st.sidebar:
        st.header("Configuration")
        st.caption(f"Connected to Pinecone index: {config['index_name']}")
        
        tab1, tab2 = st.tabs(["FinChat", "DCF Extract"])
        
        with tab1:
            company = st.selectbox(
                "Select Company", 
                company_configs["all_companies"]
            )
            
            model_option = st.radio(
                "Model",
                ["Qwen 3", "GPT-4o"],
                index=0
            )
            
            st.info(f"Model in use: {model_option}")
        
        with tab2:
            dcf_company = st.selectbox(
                "Select Company for DCF", 
                company_configs["dcf_supported_companies"]
            )
            
            dcf_model = st.radio(
                "Model for DCF Extraction",
                ["GPT-4o", "Qwen 3"],
                index=0
            )
            st.info(f"Model in use: {dcf_model}")
        

        with st.expander("Advanced Settings", expanded=False):
            k_value = st.slider("Number of documents to retrieve (k)", 1, 15, 5)
            
            namespace = company_configs["namespaces"].get(company, "")
            st.caption(f"Using namespace: {namespace}")
            
            dcf_namespace = company_configs["namespaces"].get(dcf_company, "")
            st.caption(f"DCF namespace: {dcf_namespace}")

    main_tab1, main_tab2, main_tab3 = st.tabs(["FinChat", "DCF Extract", "DCF Analyze"])
    
    with main_tab1:
        # Map namespaces and get fiscal year
        namespace = company_configs["namespaces"].get(company, "")
        fiscal_year = company_configs["get_fiscal_year"](company)
        
        predefined_queries = {
            "Revenue": f"What was {company}'s net revenue for {fiscal_year}?",
            "Operating income": f"What was {company}'s operating income for {fiscal_year}?",
            "Net Income": f"What was {company}'s net income for {fiscal_year}?",
            "Depreciation & Amortization": f"What was {company}'s depreciation and amortization for {fiscal_year}?",
            "Tax Rate": f"What was {company}'s effective tax rate for {fiscal_year}?",
            "R&D Expenses": f"What were {company}'s Research and Development expenses for {fiscal_year}?"
        }

        st.subheader(f"Extract Financial Data: {company}")
        query_type = st.radio("Question Type", ["Predefined Questions", "Custom Question"])

        if query_type == "Predefined Questions":
            selected_query = st.selectbox("Select a question", list(predefined_queries.keys()))
            user_query = predefined_queries[selected_query]
            
            with st.form(key="predefined_query_form"):
                st.text_area("Question", user_query, height=80, key="predefined_question_display")
                submit_button = st.form_submit_button("Extract Data", type="primary")
                
                if submit_button:
                    process_query = True
                else:
                    process_query = False

        else:
            with st.form(key="custom_query_form"):
                user_query = st.text_area("Enter your question", height=100, key="custom_question_input")
                submit_button = st.form_submit_button("Extract Data", type="primary")
                
                if submit_button and user_query:
                    process_query = True
                else:
                    process_query = False

        if process_query:
            with st.spinner(f"Extracting data from {company}'s 10-K..."):
                if vector_store_components and len(vector_store_components) == 3:
                    index, pc, embeddings = vector_store_components
                    
                    # Create vector store with namespace
                    vectorstore = get_vector_store(index, embeddings, namespace)
                    
                    if vectorstore:
                        # Initialize model and execute query based on selection
                        if model_option == "GPT-4o" and config["azure_api_key"]:
                            # Use GPT-4o for query processing
                            client = get_azure_gpt4o_client()
                            result = execute_financial_query_with_gpt4o(client, vectorstore, company, user_query, k_value)
                        else:
                            # Use Ollama for query processing
                            llm = get_llm()
                            prompt = create_financial_qa_prompt(company)
                            result = execute_financial_query(llm, vectorstore, prompt, user_query, k_value)
                        
                        if result:
                            # Display results
                            source_docs = result.get('source_documents', [])
                            processed_results = process_search_results(source_docs)
                            
                            # Create columns for response layout
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                # Display the results
                                st.markdown("### Answer")
                                st.markdown(result['result'])
                                    
                            with col2:
                                # Show retrieved context in a more compact format
                                st.markdown("### Source Information")
                                
                                # Show type counts
                                doc_types = {}
                                for doc in processed_results:
                                    doc_type = doc.get("type", "unknown")
                                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                                
                                for doc_type, count in doc_types.items():
                                    st.caption(f"{doc_type.capitalize()}: {count} document(s)")
                                
                                # Show page ranges
                                all_pages = []
                                for doc in processed_results:
                                    if "page_info" in doc:
                                        page_info = doc["page_info"]
                                        if "Page:" in page_info or "Pages:" in page_info:
                                            all_pages.append(page_info)
                                
                                if all_pages:
                                    st.caption("Source pages:")
                                    for page in all_pages[:5]:  # Show first 5 page references
                                        st.caption(f"• {page}")
                                    if len(all_pages) > 5:
                                        st.caption(f"...and {len(all_pages) - 5} more")
                            
                            # Display retrieved documents in expandable sections
                            st.markdown("### Source Documents")
                            display_results(processed_results)
                    else:
                        st.error("Vector store initialization failed.")
                else:
                    st.error("Could not initialize the application. Please check your configuration.")


    with main_tab2:
        st.subheader("DCF Metric Extractor")
        
        # Map namespaces and get fiscal year for DCF
        dcf_namespace = company_configs["namespaces"].get(dcf_company, "")
        dcf_fiscal_year = company_configs["get_fiscal_year"](dcf_company)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            This tool extracts all the raw financial metrics needed for DCF analysis from the company's 10-K filings.
            All metrics are provided as-is, without calculations, so you can perform your own DCF analysis with transparency.
            """)

            with st.expander("About DCF Metrics Sources", expanded=False):
                st.markdown("""
                | **Metric** | **Source in 10-K** | **Notes** |
                |------------|---------------------|-----------|
                | Revenue | Income Statement | Top line figure |
                | Operating Income | Income Statement | Also known as EBIT |
                | Depreciation & Amortization | Cash Flow Statement | Non-cash adjustment |
                | Capital Expenditures | Cash Flow Statement | Under Investing Activities |
                | Tax Information | Notes | For effective tax rate |
                """)

            generate_dcf = st.button("Generate DCF Table", type="primary", key="dcf_button")

        with col2:
            st.markdown("#### DCF Metrics")

            with st.expander("Income Statement Metrics", expanded=False):
                income_metrics = ["Revenue", "Operating Income"]
                for metric in income_metrics:
                    st.caption(f"• {metric}")

            with st.expander("Cash Flow Metrics", expanded=False):
                cashflow_metrics = ["Depreciation & Amortization", "Capital Expenditures"]
                for metric in cashflow_metrics:
                    st.caption(f"• {metric}")

            with st.expander("Other Metrics", expanded=False):
                st.caption("• Tax Rate")

        if generate_dcf:
            progress_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

            with st.spinner(f"Extracting DCF metrics from {dcf_company}'s 10-K..."):
                status_text.text("Initializing RAG system...")
                progress_bar.progress(5)

                try:
                    # Initialize the vector store components
                    if vector_store_components and len(vector_store_components) == 3:
                        index, pc, embeddings = vector_store_components
                        
                        # Create vector store with namespace for DCF
                        vectorstore = get_vector_store(index, embeddings, dcf_namespace)
                        
                        if vectorstore:
                            status_text.text("Retrieving financial documents...")
                            progress_bar.progress(15)

                            metric_contexts = company_specific_retrieval(vectorstore, dcf_company, dcf_fiscal_year)
                            status_text.text("Processing documents...")
                            progress_bar.progress(40)

                            # # Debugging
                            # with st.expander("Raw Financial Data Context", expanded=False):
                            #     context_display = ""
                            #     for metric, context in metric_contexts.items():
                            #         context_display += f"--- {metric} ---\n{context}\n\n"
                            #     st.text(context_display)
                            
                            status_text.text("Extracting financial metrics...")
                            progress_bar.progress(60)
                            
                            dcf_data = None
                            
                            if dcf_model == "GPT-4o" and config["azure_api_key"]:
                                client = get_azure_gpt4o_client()
                                dcf_data = process_with_gpt4o(client, dcf_company, metric_contexts, dcf_fiscal_year)
                            else:
                                llm = get_llm()
                                dcf_data = process_with_ollama(llm, dcf_company, metric_contexts, dcf_fiscal_year)
                            
                            if dcf_data:
                                # calculate Revenue Growth
                                dcf_data = calculate_revenue_growth(dcf_data)
                                
                                status_text.text("DCF metrics extraction complete!")
                                progress_bar.progress(100)
                                
                                st.success("Successfully extracted DCF metrics")

                                # Display JSON for reference in collapsible section
                                with st.expander("Raw JSON Data", expanded=False):
                                    st.code(json.dumps(dcf_data, indent=2))
                                
                                # Get units for display
                                units = dcf_data.get("Units", "Not specified")
                                
                                if "Revenue" in dcf_data:
                                    years = sorted(dcf_data["Revenue"].keys(), reverse=True)
                                else:
                                    years = [str(dcf_fiscal_year), str(dcf_fiscal_year-1), str(dcf_fiscal_year-2)]
                                
                                st.markdown(f"### DCF Analysis Table\n({units})")
                                
                                tab_titles = ["Extracted Metrics", "Source Documents"]
                                tabs = st.tabs(tab_titles)
                                
                                with tabs[0]:
                                    create_metrics_table(dcf_data, years, exclude_keys=["Units"])

                                with tabs[1]:
                                    create_metric_context_explorer(metric_contexts)
                                
                                st.markdown("### Export")                                
                                csv_data = prepare_csv_export(dcf_data, years)
                                # Add download button
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        "Download as CSV",
                                        csv_data,
                                        f"{dcf_company}_dcf_metrics.csv",
                                        "text/csv",
                                        key='download-csv'
                                    )
                                with col2:
                                    st.markdown("""
                                    **Next Steps:**
                                    1. Import this CSV into your financial model
                                    2. Build your DCF model with these raw metrics

                                    **Alternatively:** Upload the CSV into our DCF Analyze tool!
                                    """)

                                st.markdown(f"### Key Metrics Visualization\n({units})")                                
                                chart_df = pd.DataFrame()
                                revenue_key = "Revenue" if "Revenue" in dcf_data else next((k for k in dcf_data if "Revenue" in k), None)
                                if revenue_key:
                                    for year, value in dcf_data[revenue_key].items():
                                        if value != "N/A":
                                            try:
                                                chart_df = pd.concat([chart_df, pd.DataFrame({
                                                    'Year': [year],
                                                    'Value': [float(value)],
                                                    'Metric': [revenue_key]
                                                })])
                                            except (ValueError, TypeError):
                                                pass                                
                                income_key = "Operating_Income" if "Operating_Income" in dcf_data else next((k for k in dcf_data if "Operating" in k), None)
                                if income_key:
                                    for year, value in dcf_data[income_key].items():
                                        if value != "N/A":
                                            try:
                                                chart_df = pd.concat([chart_df, pd.DataFrame({
                                                    'Year': [year],
                                                    'Value': [float(value)],
                                                    'Metric': [income_key]
                                                })])
                                            except (ValueError, TypeError):
                                                pass
                                if not chart_df.empty:
                                    st.line_chart(chart_df, x='Year', y='Value', color='Metric')
                                
                                growth_df = pd.DataFrame()                                
                                if "Revenue_Growth" in dcf_data:
                                    for year, value in dcf_data["Revenue_Growth"].items():
                                        if value != "N/A":
                                            try:
                                                growth_df = pd.concat([growth_df, pd.DataFrame({
                                                    'Year': [year],
                                                    'Growth': [float(value)]
                                                })])
                                            except (ValueError, TypeError):
                                                pass
                                if not growth_df.empty:
                                    st.markdown("### Revenue Growth Rate")
                                    st.bar_chart(growth_df, x='Year', y='Growth')
                            else:
                                st.error("Failed to extract DCF metrics.")
                        else:
                            st.error("Vector store initialization failed.")
                    else:
                        st.error("Could not initialize the application. Please check your configuration.")
                except Exception as e:
                    st.error(f"Error generating DCF metrics: {str(e)}")

    with main_tab3:
        dcf_calculator_ui()

if __name__ == "__main__":
    main()
