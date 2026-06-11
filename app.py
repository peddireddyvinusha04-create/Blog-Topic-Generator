# app.py
import os
import json
from typing import List

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
load_dotenv()

# =====================================================
# Pydantic Schema
# =====================================================

class BlogSchema(BaseModel):
    blog_title: str
    outline_sections: List[str]
    target_audience: str
    writing_goal: str

# =====================================================
# Streamlit Page Config
# =====================================================

st.set_page_config(
    page_title="Blog Topic & Outline Generator",
    page_icon="📝",
    layout="centered"
)

# =====================================================
# UI
# =====================================================

st.title("📝 Blog Topic & Outline Generator")

st.write(
    "Generate AI-powered blog titles and structured outlines."
)

topic = st.text_input("Enter Blog Topic")

audience = st.text_input(
    "Enter Target Audience (Optional)"
)

# =====================================================
# Prompt Template
# =====================================================

prompt = PromptTemplate(
    input_variables=["topic", "audience"],
    template="""
You are an expert blog strategist.

Generate:

1. A catchy blog title
2. 5 to 7 logical outline sections
3. Target audience
4. Writing goal

TOPIC:
{topic}

AUDIENCE:
{audience}

Return ONLY valid JSON.

{{
    "blog_title": "string",
    "outline_sections": [
        "section1",
        "section2"
    ],
    "target_audience": "string",
    "writing_goal": "string"
}}
"""
)

# =====================================================
# Groq Model
# =====================================================

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

# =====================================================
# New LangChain 1.x Chain
# =====================================================

chain = prompt | llm

# =====================================================
# Generate Button
# =====================================================

if st.button("Generate Blog Outline"):

    if not topic.strip():
        st.warning("Please enter a topic.")

    else:

        if not audience.strip():
            audience = "General Readers"

        try:

            with st.spinner("Generating Blog Outline..."):

                response = chain.invoke(
                    {
                        "topic": topic,
                        "audience": audience
                    }
                )

                response_text = response.content

                result = json.loads(response_text)

                validated_output = BlogSchema(
                    **result
                )

                st.success(
                    "Blog Outline Generated Successfully!"
                )

                st.subheader("📌 Blog Title")
                st.write(
                    validated_output.blog_title
                )

                st.subheader("📚 Outline Sections")

                for idx, section in enumerate(
                    validated_output.outline_sections,
                    start=1
                ):
                    st.write(
                        f"{idx}. {section}"
                    )

                st.subheader("🎯 Target Audience")
                st.write(
                    validated_output.target_audience
                )

                st.subheader("✍️ Writing Goal")
                st.write(
                    validated_output.writing_goal
                )

                st.subheader("🧾 JSON Output")

                st.json(
                    validated_output.model_dump()
                )

        except json.JSONDecodeError:

            st.error(
                "Model returned invalid JSON."
            )

            st.write("Raw Response:")
            st.write(response_text)

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )
