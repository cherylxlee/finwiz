import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import Ollama
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
import json
import re

load_dotenv()

st.set_page_config(page_title="Financial Wizard", layout="wide")
st.title("FinWiz: Financial Data Extraction")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("INDEX_NAME")
cohere_api_key = os.getenv("COHERE_API_KEY")

with st.sidebar:
    st.header("Configuration")
    st.caption(f"Connected to Pinecone index: {index_name}")
    
    company = st.radio("Select Company", ["Apple", "Nvidia"])
    namespace = "apple_10k" if company == "Apple" else "nvidia_10k"

    st.info("Model in use: Gemma 3 12B")
    
    # Advanced settings in a collapsed expander
    with st.expander("Advanced Settings", expanded=False):
        k_value = st.slider("Number of documents to retrieve (k)", 1, 10, 5)
        st.caption(f"Using namespace: {namespace}")
        st.caption("Models hosted locally via Ollama")

def initialize_app():
    if not pinecone_api_key or not cohere_api_key or not index_name or not pinecone_environment:
        st.warning("Please make sure all environment variables are set (PINECONE_API_KEY, PINECONE_ENVIRONMENT, INDEX_NAME, COHERE_API_KEY)")
        return None, None
    
    try:
        # Setup embedding function
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_api_key
        )
        
        # Initialize Pinecone client
        pc = Pinecone(
            api_key=pinecone_api_key,
            environment=pinecone_environment
        )
        
        # List indexes to verify the one we need exists
        existing = pc.list_indexes().names()
        # st.info(f"Available Pinecone indexes: {existing}")
        
        # # Create index if it doesn't exist
        if index_name not in existing:
            st.warning(f"Index {index_name} not found. Creating it...")
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=pinecone_environment
                )
            )
        
        # # Connect to the index
        # st.info(f"Connecting to Pinecone index: {index_name}")
        index = pc.Index(index_name)
        
        # # Get index stats to see what's in there
        # stats = index.describe_index_stats()
        # st.info(f"Index stats: {stats}")
        
        # Set up the vectorstore with the existing index
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="chunk_content",
            namespace=namespace
        )
        
        return vectorstore, pc
        
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        return None, None


def get_llm(model_name="gemma3:12b"):
    return Ollama(model=model_name, base_url="http://localhost:11434")

tab1, tab2 = st.tabs(["Financial Queries", "DCF Generator"])

with tab1:
    # Predefined queries for financial analysis
    predefined_queries = {
        "Revenue Growth": f"What was {company}'s revenue growth for the most recent fiscal year compared to the previous year?",
        "Gross Profit Margin": f"What was {company}'s gross profit margin for the most recent fiscal year?",
        "Operating Margin": f"What was {company}'s operating margin for the most recent fiscal year?",
        "Net Income": f"What was {company}'s net income for the most recent fiscal year?",
        "Free Cash Flow": f"What was {company}'s free cash flow for the most recent fiscal year?",
        "Tax Rate": f"What was {company}'s effective tax rate for the most recent fiscal year?",
        "R&D Expenses": f"What were {company}'s Research and Development expenses for the most recent fiscal year?"
    }

    # Query input
    st.subheader(f"Extract Financial Data: {company}")
    query_type = st.radio("Question Type", ["Predefined Queries", "Custom Query"])

    if query_type == "Predefined Queries":
        selected_query = st.selectbox("Select a Question", list(predefined_queries.keys()))
        user_query = predefined_queries[selected_query]
        st.text_area("Question", user_query, height=80)
    else:
        user_query = st.text_area("Enter your question", height=100)

    # Execute Query Button
    execute_button = st.button("Extract Data", type="primary")

    # Results display
    if execute_button and user_query:
        # Initialize the app
        vectorstore_result = initialize_app()
        
        if vectorstore_result and len(vectorstore_result) == 2:
            vectorstore, pc = vectorstore_result
            
            # Check if vector store is empty
            if vectorstore:
                # Get the gemma model
                llm = get_llm("gemma3:12b")
                
                with st.spinner(f"Extracting data from {company}'s 10-K..."):
                    try:
                        # Create flexible prompt for all queries
                        prompt_template = """
                        You are a financial analyst extracting information from 10-K documents.
                        
                        Here is the context information from {company}'s 10-K documents:
                        
                        {context}
                        
                        Question: {question}
                        
                        Use ONLY the information in the context above to answer the question.
                        Be precise with numbers, including units (e.g., $M, %, etc.).
                        If data from multiple years is available, clearly show the trend.
                        If the information is not in the context, say "The information is not available in the provided context."
                        """
                        
                        PROMPT = PromptTemplate(
                            template=prompt_template.replace("{company}", company),
                            input_variables=["context", "question"]
                        )
                        
                        # Build the RetrievalQA chain
                        qa = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(search_kwargs={"k": k_value}),
                            chain_type_kwargs={"prompt": PROMPT},
                            return_source_documents=True  # Make sure to return source documents
                        )
                        
                        # Execute the query with RAG
                        result = qa.invoke({"query": user_query})
                        
                        # Also retrieve documents for debugging
                        test_docs = vectorstore.similarity_search(user_query, k=k_value)
                        
                        # Create columns for response layout
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Display the results
                            st.markdown("### Results")
                            st.markdown(result['result'])
                                
                        with col2:
                            # Show retrieved context in a more compact format
                            with st.expander("Retrieved Context", expanded=False):
                                if not test_docs:
                                    st.warning("No documents retrieved")
                                else:
                                    for i, doc in enumerate(test_docs):
                                        st.caption(f"Document {i+1}")
                                        st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                            
                            # More compact debug info
                            with st.expander("Debug Information", expanded=False):
                                if 'source_documents' in result:
                                    st.caption(f"Sources: {len(result['source_documents'])}")
                                else:
                                    st.caption("No source documents returned")
                                st.caption("Model used: gemma3:12b")
                    
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
            else:
                st.error("Vector store initialization failed.")
        else:
            st.error("Could not initialize the application. Please check your configuration.")


with tab2:
    st.subheader("DCF Metric Generator")
    
    # Create two columns for layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Generate key financial metrics for Discounted Cash Flow (DCF) analysis.")
        generate_dcf = st.button("Generate DCF Table", type="primary", key="dcf_button")

    with col2:
        st.markdown("#### DCF Metrics to Extract")
        st.caption("• Revenue")
        st.caption("• Gross Profit")
        st.caption("• Operating Income")
        st.caption("• Revenue Growth") 
    
    if generate_dcf:
        with st.spinner("Generating DCF metrics..."):
            try:
                # Initialize the RAG system
                vectorstore_result = initialize_app()
                
                if vectorstore_result and len(vectorstore_result) == 2:
                    vectorstore, pc = vectorstore_result
                    
                    # Two-step retrieval process
                    # Step 1: First find documents with financial data by using more specific terms
                    if company == "Apple":
                        # Target specific financial data points and table patterns
                        search_query = "Apple Total net sales Products Services Gross margin Operating income"
                    else:  # Nvidia
                        # Target specific financial data points and table patterns
                        search_query = "NVIDIA Revenue Cost of revenue Gross profit Operating income"
                    
                    # Increase k_value temporarily for this search to ensure we get the tables
                    table_k_value = min(k_value + 5, 10)  # Use more docs but cap at 10
                    
                    # Retrieve relevant documents
                    retrieved_docs = vectorstore.similarity_search(search_query, k=table_k_value)
                    
                    # Check if we found anything useful
                    found_financial_data = False
                    filtered_docs = []
                    
                    # Filter documents to find those with numeric tables
                    # Tables typically have multiple number sequences and $ symbols
                    for doc in retrieved_docs:
                        content = doc.page_content
                        # Simple heuristic: financial tables have lots of numbers and dollar signs
                        if (content.count('$') > 3 or 
                            sum(c.isdigit() for c in content) > 50 or
                            ('Revenue' in content and any(year in content for year in ['2023', '2024', '2025']))):
                            filtered_docs.append(doc)
                            found_financial_data = True
                    
                    # If we didn't find good financial data, try a different approach
                    if not found_financial_data or not filtered_docs:
                        # Step 2: Try another search with different terms
                        if company == "Apple":
                            search_query = "Apple Inc. 391,035 383,285 394,328 Net sales Products Services"
                        else:  # Nvidia
                            search_query = "NVIDIA Corporation 130,497 60,922 26,974 Revenue"
                        
                        retrieved_docs = vectorstore.similarity_search(search_query, k=table_k_value)
                        filtered_docs = retrieved_docs  # Use all docs from second search
                    
                    if filtered_docs:
                        # Combine document content
                        context_text = "\n\n".join([doc.page_content for doc in filtered_docs])
                        st.info(f"Retrieved {len(filtered_docs)} documents with financial data")
                        
                        # Display raw context to help with debugging (can be removed in production)
                        with st.expander("Raw Financial Data Context", expanded=False):
                            st.text(context_text)
                        
                        # Get the Gemma model
                        llm = get_llm("gemma3:12b")
                        
                        # Create an improved prompt that's more direct and specific
                        if company == "Apple":
                            dcf_prompt = f"""
                            You are FinancialGPT, a financial data extraction expert.
                            
                            TASK:
                            Extract Apple's financial metrics from the financial data provided.
                            
                            LOOK FOR THIS DATA:
                            1. First find the table with Apple's revenue or net sales figures
                            2. Extract these exact metrics for the 3 most recent years:
                               - Total net sales or Revenue (look for rows with "Total net sales")
                               - Gross profit (look for rows with "Gross margin")
                               - Operating income (look for rows with "Operating income")
                            3. Note the units (should be in millions) from any header text like "In millions"
                            
                            DATA FORMATTING RULES:
                            - Extract plain numbers WITHOUT commas or currency symbols
                            - Do NOT calculate any growth rates
                            - If you can't find a value, use "N/A"
                            
                            FINANCIAL DATA:
                            {context_text}
                            
                            OUTPUT FORMAT (JSON):
                            {{
                              "Units": "In millions (or whatever unit is specified)",
                              "Revenue": {{
                                "2024": [value],
                                "2023": [value],
                                "2022": [value]
                              }},
                              "Gross Profit": {{
                                "2024": [value],
                                "2023": [value],
                                "2022": [value]
                              }},
                              "Operating Income": {{
                                "2024": [value],
                                "2023": [value],
                                "2022": [value]
                              }}
                            }}
                            
                            Output ONLY the JSON, no explanations.
                            """
                        else:  # Nvidia
                            dcf_prompt = f"""
                            You are FinancialGPT, a financial data extraction expert.
                            
                            TASK:
                            Extract NVIDIA's financial metrics from the financial data provided.
                            
                            LOOK FOR THIS DATA:
                            1. First find the table with NVIDIA's revenue figures
                            2. Extract these exact metrics for the 3 most recent years:
                               - Revenue (look for rows with "Revenue")
                               - Gross profit (look for rows with "Gross profit")
                               - Operating income (look for rows with "Operating income")
                            3. Note the units (should be in millions) from any header text like "In millions"
                            
                            DATA FORMATTING RULES:
                            - Extract plain numbers WITHOUT commas or currency symbols
                            - Do NOT calculate any growth rates
                            - If you can't find a value, use "N/A"
                            
                            FINANCIAL DATA:
                            {context_text}
                            
                            OUTPUT FORMAT (JSON):
                            {{
                              "Units": "In millions (or whatever unit is specified)",
                              "Revenue": {{
                                "2025": [value],
                                "2024": [value],
                                "2023": [value]
                              }},
                              "Gross Profit": {{
                                "2025": [value],
                                "2024": [value],
                                "2023": [value]
                              }},
                              "Operating Income": {{
                                "2025": [value],
                                "2024": [value],
                                "2023": [value]
                              }}
                            }}
                            
                            Output ONLY the JSON, no explanations.
                            """
                        
                        # Call the model
                        response = llm.invoke(dcf_prompt)
                        
                        # Extract JSON from response
                        json_pattern = r'\{[\s\S]*\}'
                        json_match = re.search(json_pattern, response)
                        
                        if json_match:
                            try:
                                json_str = json_match.group(0)
                                dcf_data = json.loads(json_str)
                                
                                # Extract the units for display
                                units = dcf_data.get("Units", "Not specified")
                                
                                # Calculate Revenue Growth based on the Revenue data
                                if "Revenue" in dcf_data:
                                    years = sorted(dcf_data["Revenue"].keys(), reverse=True)
                                    
                                    # Make sure we have at least 2 years of data to calculate growth
                                    if len(years) >= 2:
                                        # Add Revenue Growth to the JSON
                                        dcf_data["Revenue Growth (%)"] = {}
                                        
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
                                                        dcf_data["Revenue Growth (%)"][current_year] = round(growth, 2)
                                                    else:
                                                        dcf_data["Revenue Growth (%)"][current_year] = "N/A"
                                                except (ValueError, TypeError):
                                                    dcf_data["Revenue Growth (%)"][current_year] = "N/A"
                                            else:
                                                dcf_data["Revenue Growth (%)"][current_year] = "N/A"
                                        
                                        # Add N/A for the earliest year since we can't calculate growth
                                        dcf_data["Revenue Growth (%)"][years[-1]] = "N/A"
                                
                                # Display JSON for reference in collapsible section
                                with st.expander("Raw JSON Data", expanded=False):
                                    st.code(json.dumps(dcf_data, indent=2))
                                
                                # Transform to DataFrame for visualization
                                st.markdown(f"### DCF Analysis Table\n({units})")
                                
                                # Create a table with metrics as rows and years as columns
                                # Filter out the "Units" key from metrics
                                metrics = [key for key in dcf_data.keys() if key != "Units"]
                                
                                # Get all years across all metrics
                                all_years = set()
                                for metric in metrics:
                                    all_years.update(dcf_data[metric].keys())
                                years = sorted(list(all_years), reverse=True)  # Sort years in descending order
                                
                                # Create DataFrame with proper formatting
                                data = []
                                for metric in metrics:
                                    row = [metric]
                                    for year in years:
                                        value = dcf_data[metric].get(year, "N/A")
                                        # For display, format large numbers with commas and add % to growth rates
                                        if metric == "Revenue Growth (%)" and value != "N/A":
                                            row.append(f"{value}%")
                                        elif isinstance(value, (int, float)) and metric != "Revenue Growth (%)":
                                            row.append(f"{value:,}")
                                        else:
                                            row.append(value)
                                    data.append(row)
                                
                                df = pd.DataFrame(data, columns=["Metric"] + years)
                                st.table(df)
                                
                                # Add a download button
                                # For CSV, keep the raw values without formatting
                                data_raw = []
                                for metric in metrics:
                                    row = [metric]
                                    for year in years:
                                        value = dcf_data[metric].get(year, "N/A")
                                        row.append(value)
                                    data_raw.append(row)
                                
                                df_raw = pd.DataFrame(data_raw, columns=["Metric"] + years)
                                csv = df_raw.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download DCF Data as CSV",
                                    csv,
                                    "dcf_metrics.csv",
                                    "text/csv",
                                    key='download-csv'
                                )
                                
                                # DCF Visualization
                                st.markdown(f"### Key Metrics Visualization\n({units})")
                                
                                # Convert data for charting - use raw values
                                chart_df = pd.DataFrame()
                                
                                # Add Revenue
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
                                
                                # Add Operating Income
                                income_key = "Operating Income" if "Operating Income" in dcf_data else next((k for k in dcf_data if "Operating" in k), None)
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
                                
                                # Create a separate chart for Revenue Growth
                                growth_df = pd.DataFrame()
                                
                                if "Revenue Growth (%)" in dcf_data:
                                    for year, value in dcf_data["Revenue Growth (%)"].items():
                                        if value != "N/A":
                                            try:
                                                growth_df = pd.concat([growth_df, pd.DataFrame({
                                                    'Year': [year],
                                                    'Growth': [float(value)]
                                                })])
                                            except (ValueError, TypeError):
                                                pass
                                
                                if not growth_df.empty:
                                    st.markdown("### Revenue Growth Rate (%)")
                                    st.bar_chart(growth_df, x='Year', y='Growth')
                                
                            except Exception as e:
                                st.error(f"Error processing financial data: {str(e)}")
                                st.code(response)
                        else:
                            st.warning("Could not detect JSON format in response.")
                            st.code(response)
                    else:
                        st.error("Could not find the financial statement tables in the documents. Try increasing the k-value in settings.")
                else:
                    st.error("Could not initialize RAG system. Please check your configuration.")
            
            except Exception as e:
                st.error(f"Error generating DCF metrics: {str(e)}")
