import streamlit as st

from analysis import (
    PROVIDER_GROQ,
    PROVIDER_OPENAI,
    MissingAPIKeyError,
    analyze_text,
    as_percent,
    resolve_api_key,
)

st.set_page_config(page_title="BabydollAI Gateway", page_icon="🎭", layout="centered")
st.title("🎭 BabydollAI: Enterprise Onboarding Gateway")
st.write(
    "Forensic behavioral screening middleware engineered via OpenAI Codex and GPT-5.6 infrastructure."
)

st.sidebar.header("🔑 Authentication Framework")
provider_label = st.sidebar.selectbox(
    "Select Inference Backbone:",
    [
        "Groq Llama 3.3 70B (Free Public Tier)",
        "OpenAI GPT-5.6 (Official Tier)",
    ],
)
user_key = st.sidebar.text_input(
    "Enter Personal API Key (Optional Override):",
    type="password",
)

st.subheader("Onboarding Evaluation Prompt:")
user_input = st.text_area(
    "Evaluation Scenario: 'What does a healthy boundary look like to you, and how do you handle it when a partner brings up a mistake you made?'",
    placeholder="Type response to analyze...",
)

if st.button("Execute Deep Behavioral Analysis"):
    if not user_input:
        st.warning("Please input text to execute analysis.")
    else:
        provider = (
            PROVIDER_OPENAI
            if provider_label.startswith("OpenAI")
            else PROVIDER_GROQ
        )
        secret_name = "OPENAI_API_KEY" if provider == PROVIDER_OPENAI else "GROQ_API_KEY"
        api_key = resolve_api_key(user_key, secret_name)

        with st.spinner("Executing forensic linguistic processing..."):
            try:
                parsed_result = analyze_text(
                    user_input,
                    provider=provider,
                    api_key=api_key,
                )
            except MissingAPIKeyError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Inference Connection Exception: {exc}")
            else:
                st.success("Linguistic Analysis Complete!")

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Secure Alignment",
                    f"{as_percent(parsed_result.get('secure_rating', 0))}%",
                )
                col2.metric(
                    "Covert Avoidance",
                    f"{as_percent(parsed_result.get('avoidant_rating', 0))}%",
                )
                col3.metric(
                    "Toxic Risk Score",
                    f"{as_percent(parsed_result.get('toxic_risk_rating', 0))}%",
                )

                st.write("### 🚨 Camouflage & Deception Index")
                st.progress(as_percent(parsed_result.get("deception_score", 0)) / 100)

                keywords = parsed_result.get("flagged_keywords", [])
                if keywords:
                    st.write(
                        "**Weaponized Terms Detected:** "
                        + ", ".join([f"`{k}`" for k in keywords])
                    )

                st.info(f"**Gatekeeper Action:** {parsed_result.get('status', 'PENDING')}")
                st.write(
                    f"**Psychological Rationale:** {parsed_result.get('justification')}"
                )
                st.success(
                    f'💡 **Secure Realignment:** "{parsed_result.get("secure_translation")}"'
                )
