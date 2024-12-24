# AI-Powered Medical Assistant

This repository contains the backend implementation of an **AI-Powered Medical Assistant** leveraging Generative AI. The assistant takes user inputs (symptoms) and provides diagnostics, treatment recommendations, and generates a professional PDF report summarizing the findings.

## Features

1. **Symptom Collection**: Gathers detailed information about the user's symptoms.
2. **Diagnostics**: Analyzes the symptoms and suggests potential conditions and next steps.
3. **Treatment Planning**: Recommends specific treatment plans, including lifestyle changes and when to seek medical help.
4. **Summary Generation**: Creates an empathetic, user-friendly summary of the diagnostic process and findings.
5. **PDF Report Generation**: Generates a professional report in PDF format summarizing all information.

## System Architecture

The application is structured using multiple agents, each responsible for a specific task:

1. **Symptom Collector Agent**: Gathers the user's symptoms and organizes them.
2. **Diagnostic Agent**: Analyzes symptoms to identify potential conditions.
3. **Treatment Agent**: Suggests treatment plans tailored to the user.
4. **Summary Agent**: Summarizes the diagnostics and recommendations.
5. **Document Generation Agent**: Generates a structured PDF report for the user.

### Task Workflow

- **Symptom Collection** → **Diagnostics** → **Treatment Planning** → **Summary Creation** → **PDF Report Generation**

### Technologies Used

- **Python**: Core programming language.
- **Generative AI**: Used for natural language processing and generating diagnostic insights.
- **ReportLab**: For generating PDF documents.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - `reportlab`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-medical-assistant.git
   cd ai-medical-assistant
