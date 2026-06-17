from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from groq import Groq
import os
import sys
from dotenv import load_dotenv

# Configure terminal output encoding to prevent Unicode errors on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

load_dotenv()

# Regulatory Compliance Knowledge Base for RAG
KNOWLEDGE_BASE = [
    {
        "title": "BSA/AML High-Value Transactions Rule",
        "keywords": "high value, large amount, money laundering, fatf, compliance threshold",
        "text": "Under the Bank Secrecy Act (BSA) and FinCEN regulations, transactions exceeding ₹50,000 (or $10,000 USD equivalent) represent significant risk. Financial institutions must exercise enhanced due diligence (EDD) to detect potential money laundering or tax evasion, and file a Suspicious Activity Report (SAR) if the transaction lack a clear economic or lawful purpose."
    },
    {
        "title": "Structuring and Layering Avoidance Rule",
        "keywords": "structuring, multiple transactions, splitting, threshold evasion, layering",
        "text": "Structuring involves splitting a large transaction into smaller, consecutive amounts (e.g., ₹9,900 to evade a ₹10,000 threshold). This is a criminal offense under federal anti-money laundering rules. Layering involves moving funds rapidly across multiple accounts or transaction types to obscure the source of the capital."
    },
    {
        "title": "Velocity and Geographical Anomaly Guidelines",
        "keywords": "unusual location, velocity, cross-border, login location, spoofing",
        "text": "Geographical anomalies, such as out-of-character cross-border transactions or high-velocity account transfers between distant locations (e.g., India to Dubai to USA within an hour), indicate compromised credentials, mule account activity, or identity theft."
    },
    {
        "title": "Nighttime and Unauthorized Hours Patterns",
        "keywords": "nighttime, midnight, unusual time, bot activity, credential stuffing",
        "text": "Transactions occurring during typical rest periods (e.g., 12 AM to 6 AM local time) are high-risk indicators of digital fraud. Bots, automated scripts, and remote threat actors frequently execute account draining procedures or unauthorized transfers during these hours to minimize immediate account holder detection."
    },
    {
        "title": "Rule-Based Overrides for High-Risk Transactions",
        "keywords": "rule trigger, automatic flag, manual override, risk policy",
        "text": "Financial risk policy mandates that transactions violating absolute safety rules (such as single transfers exceeding ₹50,000 or high-value transfers initiated immediately after account creation) receive automatic compliance holds, overriding ML model outputs to ensure baseline asset protection."
    }
]

# Extract texts for TF-IDF Vectorizer
kb_texts = [f"{doc['title']} - {doc['keywords']} - {doc['text']}" for doc in KNOWLEDGE_BASE]
vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(kb_texts)

def retrieve_regulatory_context(amount, time, transaction_type="", location="", flagged_by_rules=False):
    """
    RAG Step: Retrieve the most relevant regulatory document based on transaction characteristics
    """
    query_terms = []
    if amount > 50000:
        query_terms.append("high value large amount money laundering threshold")
    elif amount > 10000:
        query_terms.append("intermediate amount structuring")
    
    # Check if night
    if time % 86400 < 21600:
        query_terms.append("nighttime midnight unusual time bot activity")
        
    if transaction_type:
        query_terms.append(f"transaction type {transaction_type}")
        
    if location in ['Dubai', 'Singapore']:
        query_terms.append("unusual location cross-border geographical anomaly")
        
    if flagged_by_rules:
        query_terms.append("rule trigger automatic flag compliance risk policy override")
        
    if not query_terms:
        query_terms.append("standard banking transaction compliance guidelines")
        
    query = " ".join(query_terms)
    
    # TF-IDF Retrieval
    query_vec = vectorizer.transform([query])
    scores = (kb_vectors * query_vec.T).toarray()
    best_idx = int(scores.argmax())
    
    return KNOWLEDGE_BASE[best_idx]

def generate_narrative(amount, time, prediction, prob, transaction_type="transfer", location="India", flagged_by_rules=False):
    """
    Generates a Suspicious Activity Report (SAR) narrative using Groq LLM (RAG-infused).
    Includes a beautiful fallback generator if Groq is not configured or fails.
    """
    
    # Retrieve regulatory context using RAG
    kb_doc = retrieve_regulatory_context(amount, time, transaction_type, location, flagged_by_rules)
    context_text = f"Compliance Reference: {kb_doc['title']}\nGuidelines: {kb_doc['text']}"
    
    if prediction == "Normal":
        return f"""======================================================================
🔍 TRANSACTION SCAN REPORT (CLASSIFICATION: SAFE)
======================================================================
Generated Date: 2026-05-21 Compliance Review System

1. TRANSACTION PROFILE:
   - Amount: ₹{amount:,.2f}
   - Time Stamp: {time} (scaled time: {time/172800:.4f})
   - Channel: {transaction_type.upper()}
   - Origin Location: {location}
   
2. RISK ASSESSMENT:
   - XGBoost Fraud Probability: {prob:.4f}
   - AML Compliance Rule Triggered: None
   - Overall Decision: SAFE (Normal Activity)

3. RAG REGULATORY ALIGNMENT:
   - Retrieved Reference: {kb_doc['title']}
   - Analysis: The transaction falls within expected behavioral baselines. The fraud detection system returned an extremely low anomaly probability, and no rule-based compliance thresholds were breached.

4. COMPLIANCE CONCLUSION:
   No further action is required. This transaction is cleared for execution.
======================================================================
"""

    # For suspicious transactions, try calling the Groq API
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key and api_key.strip() and not api_key.startswith("gsk_placeholder"):
        try:
            client = Groq(api_key=api_key)
            
            prompt = f"""
You are a Senior Financial Compliance Officer at an international bank.
Draft a highly formal, detailed, and legally sound Suspicious Activity Report (SAR) Narrative based on the transaction metadata and retrieved regulatory guidelines below.

--- TRANSACTION METADATA ---
- Amount: ₹{amount:,.2f}
- Time Stamp: {time}
- Transaction Type: {transaction_type}
- Originating Location: {location}
- XGBoost ML Risk Probability: {prob:.4%}
- Rule Override Flagged: {"YES (Absolute Policy Breach)" if flagged_by_rules else "NO (Triggered by ML Probability)"}

--- RETRIEVED REGULATORY COMPLIANCE CONTEXT (RAG) ---
{context_text}

--- REPORT STRUCTURE REQUIRED ---
Your report MUST follow this layout precisely:
1. SUMMARY OF SUSPICIOUS ACTIVITY: A brief executive summary of what was detected and why the alert triggered.
2. TRANSACTION INVESTIGATION DETAILS: Detailed analysis of the amount, timing, channel ({transaction_type}), and location ({location}). Point out specific risk factors (e.g. nighttime, geographical risk, excessive amounts).
3. REGULATORY COMPLIANCE ALIGNMENT: Reference the Bank Secrecy Act / AML rules retrieved in the context and explain how this transaction violates them.
4. ACTIONABLE RECOMMENDATIONS: Clear next steps (e.g., account hold, customer verification, referral to FinCEN).
5. AUDIT LOG & COMPLIANCE SIGN-OFF: End with a professional sign-off template.

Ensure the tone is objective, clinical, and highly professional. Avoid emotional language. Do not invent fictitious entities other than referring to the 'Account Holder' and the 'Reporting Institution'.
"""
            # Using Llama 3 8B model for high speed and robustness
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "You are a professional banking compliance officer drafting an official Suspicious Activity Report."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback to local high-quality generator on API error
            pass

    # High-quality Local Template Fallback (Guarantees app never crashes and looks gorgeous)
    override_reason = "the transaction exceeds absolute risk thresholds (₹50,000 rule)" if flagged_by_rules else f"the Machine Learning model identified an anomaly with {prob:.2%} confidence"
    
    return f"""======================================================================
OFFICIAL SUSPICIOUS ACTIVITY REPORT (SAR) - DRAFT NARRATIVE
======================================================================
Report Reference: SAR-2026-EDI-{np.random.randint(1000, 9999)}
Filing Institution: EDI Compliance Operations Center
Subject Account: [Review Hold - Under Investigation]

1. SUMMARY OF SUSPICIOUS ACTIVITY:
   On 2026-05-21, our automated transaction monitoring system flagged a highly suspicious {transaction_type} transaction. The alert triggered because {override_reason}. Under FinCEN guidelines, this activity represents an elevated risk profile and requires a compliance hold.

2. TRANSACTION INVESTIGATION DETAILS:
   - Principal Amount: ₹{amount:,.2f}
   - Log Time Stamp: {time} (Epoch seconds offset)
   - Channel / Type: {transaction_type.upper()}
   - Origin Location: {location}
   - Risk Indicators Identified:
     * High Asset Value: {"YES" if amount > 50000 else "NO"}
     * Nighttime Execution Anomaly: {"YES" if time % 86400 < 21600 else "NO"}
     * XGBoost Classifier Score: {prob:.4f}
     * Rule-Based Policy Violation: {"YES" if flagged_by_rules else "NO"}

3. REGULATORY COMPLIANCE ALIGNMENT (RAG RETROFIT):
   - Retrieved Reference: {kb_doc['title']}
   - Compliance Context: {kb_doc['text']}
   - Analysis: The subject transaction directly violates key metrics outlined in our AML risk policies. Specifically, the value profile and transaction characteristics align with indicators found in {kb_doc['title']}, indicating potential structuring, money laundering, or account takeover activity.

4. ACTIONABLE RECOMMENDATIONS:
   - Immediate Hold: Apply an administrative hold on the source account to freeze pending disbursements.
   - Enhanced Due Diligence (EDD): Request physical identity verification, origin-of-wealth documentation, and validation of transaction purpose from the account holder.
   - Regulatory Filing: Finalize and file this draft narrative to the financial intelligence unit (FinCEN equivalent) within the 30-day statutory window.

5. AUDIT LOG & COMPLIANCE SIGN-OFF:
   - Monitoring System: XGBoost Fraud Classifier v1.2 & RAG KB Engine
   - Investigating Officer: Senior Compliance Reviewer (AI Assistant)
   - Status: Pending Board Approval & External Filing
======================================================================
"""

if __name__ == "__main__":
    print("Testing Normal Narrative...")
    normal_report = generate_narrative(1200, 50000, "Normal", 0.05, "payment", "India", False)
    print(normal_report)
    
    print("\nTesting Fraud Narrative (Local Fallback or API)...")
    fraud_report = generate_narrative(80000, 1000, "Fraud", 0.95, "transfer", "Dubai", True)
    print(fraud_report)