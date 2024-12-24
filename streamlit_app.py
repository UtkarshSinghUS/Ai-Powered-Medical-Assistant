
import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, TXTSearchTool, PDFSearchTool
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from datetime import datetime
from io import BytesIO
from fpdf import FPDF

# Set environment variables
os.environ["GROQ_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "695932c00558f2f56996ba136c6bac3271e91e7a"
os.environ["COHERE_API_KEY"] = "nFV7Lwho5KS7lWQwKKWVTqa3JeXk2on8sesgbkAv"

# Initialize Language Model
llm = ChatGroq(model="groq/llama-3.1-70b-versatile", api_key=os.environ["GROQ_API_KEY"])



# # Define PDF generation function
# def generate_pdf(output_text, file_name="Medical_Report.pdf"):
#     doc = SimpleDocTemplate(file_name, pagesize=letter)
#     styles = getSampleStyleSheet()
#     style = styles['Normal']
#     content = [Paragraph(line, style) for line in output_text.split("\n") if line.strip()]
#     doc.build(content)
#     return file_name

# # Define PDF generation function
# def generate_pdf(output_text, file_name=None):
#     try:
#         if not file_name:
#             file_name = f"Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
#         doc = SimpleDocTemplate(file_name, pagesize=letter)
#         styles = getSampleStyleSheet()
#         style = styles['Normal']
#         content = [Paragraph(line, style) for line in output_text.split("\n") if line.strip()]
#         doc.build(content)
#         return file_name
#     except Exception as e:
#         st.error(f"Error generating PDF: {str(e)}")
#         return None


# # Streamlit App Title and Description
# st.title("AI-Powered Medical Assistant")
# st.write("A diagnostic tool to help analyze symptoms, provide potential diagnoses, and suggest treatments.")

# Tools
medical_pdf_tool = PDFSearchTool(
    pdf_path='medical_references.pdf',
    config={
        "llm": {"provider": "groq", "config": {"model": "groq/mixtral-8x7b-32768"}},
        "embedder": {
            "provider": "cohere",
            "config": {"model": "embed-english-v3.0", "api_key": os.environ["COHERE_API_KEY"]},
        },
    },
)

medical_txt_tool = TXTSearchTool(
    txt='medical_data.txt',
    config={
        "llm": {"provider": "groq", "config": {"model": "groq/mixtral-8x7b-32768"}},
        "embedder": {
            "provider": "cohere",
            "config": {"model": "embed-english-v3.0", "api_key": os.environ["COHERE_API_KEY"]},
        },
    },
)

# Updated Agents
symptom_collector_agent = Agent(
    role="Symptom Collector",
    goal="Collect detailed symptoms from the user for diagnostic purposes,it should give symptoms only from the user given symptoms from basic_symptoms , location , duration , severity , pattern , additional_factors and nothing else.",
    backstory="A patient-centric assistant skilled in gathering comprehensive medical details for accurate analysis.",
    expected_output="A well-structured list of symptoms,it should give symptoms only from the user given symptoms from basic_symptoms , location , duration , severity , pattern , additional_factors and nothing else, duration mentioned by the user.",
    verbose=True,
    max_iter=1,
    llm=llm
)

diagnostic_agent = Agent(
    role="Diagnostic Specialist",
    goal="Analyze the user's symptoms,it should give symptoms only from the user given symptoms from basic symptoms , location , duration , severity , pattern , additional_factors and nothing else, duration mentioned by the user to identify potential conditions. Provide specific recommendations for further tests, treatments, or specialist consultations.",
    backstory="An expert in diagnostics, capable of interpreting symptoms and guiding the patient with clear steps for recovery or further evaluation.",
    expected_output=(
        "A likely diagnosis based on the symptoms,it should give symptoms only from the user given symptoms from basic symptoms ,location , duration , severity , pattern , additional_factors and nothing else. Include specific next steps, such as recommended diagnostic tests, lifestyle changes, medications, or consulting a specialist."
    ),
    verbose=True,
    max_iter=1,
    llm=llm
)

treatment_agent = Agent(
    role="Treatment Advisor",
    goal="Provide actionable and patient-specific treatment recommendations based on the diagnosis. Ensure all advice is practical and safe.",
    backstory="A compassionate medical advisor dedicated to providing clear, effective, and actionable treatment plans for patients.",
    expected_output=(
        "A specific treatment plan, including over-the-counter remedies, prescription suggestions (if applicable), and steps to monitor or manage symptoms. Highlight when a doctor visit is essential."
    ),
    verbose=True,
    max_iter=1,
    llm=llm
)

summary_agent = Agent(
    role="Summary Specialist",
    goal="Generate a concise, empathetic, and user-friendly summary of the diagnostic findings, identified conditions, and actionable next steps. Ensure the summary is clear and easy to understand.",
    backstory=(
        "A compassionate and understanding medical expert who excels at summarizing complex medical information in a way that is approachable and comprehensible for patients. "
        "This agent's primary focus is to distill critical insights into a brief and clear summary, providing patients with actionable next steps and recommendations."
    ),
    expected_output=(
        "A clear and concise summary of the diagnostic process, including identified conditions, key diagnostic insights, and actionable next steps such as treatments, tests, or lifestyle adjustments."
    ),
    verbose=True,
    max_iter=1,
    llm=llm
)


document_generation_agent = Agent(
    role="Document Creator",
    goal="Generate a detailed and well-structured document summarizing the diagnostic findings, recommendations, and other relevant information for the user.",
    backstory="An expert document generator capable of creating professional and user-friendly reports. Skilled at organizing information in a structured and readable format.",
    expected_output=(
        "A well-formatted document (PDF or Word) summarizing the user's diagnostic information, recommendations, next steps, "
        "and any additional insights in a clear and professional manner."
    ),
    verbose=True,
    max_iter=1,
    llm=llm
)

symptom_collection_task = Task(
    description="Collect detailed symptoms,it should give symptoms only from the user given symptoms from basic symptoms , location , duration , severity , pattern , additional_factors and nothing else, duration from the user to prepare for diagnosis.",
    expected_output="Structured details of symptoms,it should give symptoms only from the user given symptoms from basic symptoms , location , duration , severity , pattern , additional_factors and nothing else, duration.",
    agent=symptom_collector_agent,
    tools=[],
    # human_input=True
)

diagnostic_task = Task(
    description=(
        "Using the collected symptoms,it should give symptoms only from the user given symptoms from basic symptoms , location , duration , severity , pattern , additional_factors and nothing else identify potential conditions the user may have. "
        "Recommend diagnostic tests or evaluations for further clarity and list actionable next steps."
    ),
    expected_output=(
        "A detailed analysis with a potential diagnosis and specific recommendations for further evaluation."
    ),
    agent=diagnostic_agent,
    context=[symptom_collection_task]
    # human_input=True
)

treatment_task = Task(
    description=(
        "Based on the diagnosis, create a patient-specific treatment plan. Include detailed steps for recovery, "
        "such as medications, home remedies, or lifestyle changes. Highlight situations requiring immediate medical attention."
    ),
    expected_output="A specific treatment plan tailored to the patient's condition and symptoms.",
    agent=treatment_agent,
    context=[diagnostic_task]
    # human_input=True
)

summary_task = Task(
    description=(
        "Use the findings and recommendations provided by the Diagnostic Specialist agent to create a concise and user-friendly summary. "
        "The summary should include the identified condition (if any), key diagnostic insights, suggested next steps, and any other important details. "
        "Ensure that the summary is empathetic and easy for the user to understand."
    ),
    expected_output=(
        "A brief and clear summary of the diagnostic process, including the findings, likely condition(s), and actionable next steps "
        "such as recommended treatments, tests, or lifestyle adjustments."
    ),
    agent=summary_agent,
    context=[symptom_collection_task,diagnostic_task,treatment_task],  # Add any necessary tasks or agents that provide context for the summary
    tools=[],    # Add tools if the summary task requires access to additional resources
    verbose=True
    # human_input=True
)

document_generation_task = Task(
    description=(
        "Create a comprehensive and structured document summarizing the user's diagnostic information, medical history, identified condition(s), "
        "recommendations, and next steps. The document should include sections like 'Introduction', 'Summary of Findings', 'Recommendations', "
        "'Next Steps', and 'Additional Information'. Ensure the document is formatted professionally for easy readability."
    ),
    expected_output=(
        "A document (PDF or Word) with the following sections: "
        "1. Introduction\n"
        "2. Summary of Findings\n"
        "3. Recommendations\n"
        "4. Next Steps\n"
        "5. Additional Information\n"
        "The document should be well-structured, professional, and easy to read."
    ),
    agent=document_generation_agent,
    context=[symptom_collection_task,diagnostic_task,treatment_task,summary_task],  # Add any necessary tasks or agents that provide input for the document
    tools=[],
    verbose=True
    # human_input=True
)



health_assistant_crew = Crew(
    agents=[
        symptom_collector_agent,
        diagnostic_agent,
        treatment_agent,
        summary_agent,
        document_generation_agent
    ],
    tasks=[
        symptom_collection_task,
        diagnostic_task,
        treatment_task,
        summary_task,
        document_generation_task
    ],
    process=Process.sequential,
    verbose=True,

)



from fpdf import FPDF
from io import BytesIO

def generate_pdf(content):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Adding content to PDF
    pdf.multi_cell(0, 10, content)

    # Create a BytesIO buffer
    pdf_buffer = BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1')  # Generate PDF as string
    pdf_buffer.write(pdf_output)  # Write the output to the buffer
    pdf_buffer.seek(0)  # Reset buffer pointer to the start
    return pdf_buffer





import streamlit as st

# Streamlit App Title and Description
st.title("AI-Powered Medical Assistant")
st.write("A diagnostic tool to help analyze symptoms, provide potential diagnoses, and suggest treatments.")

# Form for Symptom Collection
with st.form(key="symptom_form"):
    st.subheader("Tell me your basic symptoms:")
    
    # Collecting basic symptom description
    basic_symptoms = st.text_area("Describe your symptoms in detail:")
    
    # Additional details
    st.write("Provide specific details below:")
    
    location = st.text_input("Location (e.g., chest, head, abdomen):")
    duration = st.text_input("Duration (e.g., 2 days, 1 week):")
    severity = st.select_slider("Severity (1 = Mild, 10 = Severe):", options=range(1, 11))
    pattern = st.text_input("Pattern (e.g., intermittent, continuous):")
    additional_factors = st.text_area("Additional factors (e.g., triggers, previous conditions):")
    
    # Submit button
    submitted = st.form_submit_button("Submit Symptoms")



if submitted:
    if basic_symptoms and location and duration:
        st.success("Symptoms submitted successfully!")
        
        # Prepare the input for processing
        user_symptom_data = f"""
        Basic Symptoms: {basic_symptoms}
        Location: {location}
        Duration: {duration}
        Severity: {severity}
        Pattern: {pattern}
        Additional Factors: {additional_factors}
        """

# Simulating AI processing (replace with actual assistant call)
        st.spinner("Analyzing symptoms, please wait...")
        diagnostic_summary = f"Diagnostic Summary\n\nBased on the provided symptoms:\n\n{user_symptom_data}"

        # Display Results
        st.subheader("Diagnostic Summary")
        st.write(diagnostic_summary)

        
        # Run the crew
        with st.spinner("Analyzing symptoms, please wait..."):
            result = health_assistant_crew.kickoff(inputs={"user_symptom_data": user_symptom_data})

        if result and result.raw:
            st.success("Analysis Complete!")

            # Display Results
            st.subheader("Diagnostic Summary")
            with st.expander("View Full Report"):
                st.write(result.raw)

        pdf_content = f"Medical Report\n\n{result.raw}"
        
        # Create the PDF buffer
        pdf_buffer = generate_pdf(pdf_content)

        # Add Download Button
        st.download_button(
            label="Download Medical Report (PDF)",
            data=pdf_buffer,
            file_name=f"Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
         )


  
