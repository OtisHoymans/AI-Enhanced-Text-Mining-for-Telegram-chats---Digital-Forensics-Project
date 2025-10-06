import streamlit as st
import json
import pandas as pd
import re
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
from typing import List, Dict
from collections import Counter
import numpy as np
from config import API_KEY
import os

# Page Configuration
st.set_page_config(
    page_title="Telegram Forensic Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Include Font Awesome for icons
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css">',
    unsafe_allow_html=True
)

# Load external CSS file
css_path = os.path.join("styles.css")
try:
    with open(css_path, "r") as css_file:
        css = css_file.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.error("CSS file not found. Please ensure 'styles.css' exists.")
    st.stop()

# Analyzer Class
class TelegramForensicAnalyzer:
    def __init__(self, api_key: str = API_KEY):
        genai.configure(api_key=api_key)
        for name in ["models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-2.0-pro-exp"]:
            try:
                m = genai.GenerativeModel(name)
                m.generate_content("Test")
                self.model = m
                self.model_name = name
                break
            except:
                continue
        if not hasattr(self, "model"):
            raise RuntimeError("Gemini model not available. Check API key.")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["hour"] = df["date"].dt.hour
            df["day"] = df["date"].dt.date
            df["day_of_week"] = df["date"].dt.day_name()
            df["month"] = df["date"].dt.to_period('M')
        df["text"] = df["text"].astype(str).fillna("")
        df["msg_length"] = df["text"].apply(len)
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
        df["has_url"] = df["text"].str.contains(r'http[s]?://', regex=True, na=False)
        df["has_phone"] = df["text"].str.contains(r'\+?\d{10,}', regex=True, na=False)
        df["has_email"] = df["text"].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True, na=False)
        
        # Calculate response times
        df = df.sort_values('date').reset_index(drop=True)
        df['time_since_prev'] = df.groupby('from')['date'].diff().dt.total_seconds() / 60  # minutes
        
        return df

    def extract_urls(self, df: pd.DataFrame) -> List[str]:
        """Extract all URLs from messages"""
        urls = []
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        for text in df['text']:
            found = re.findall(url_pattern, str(text))
            urls.extend(found)
        return list(set(urls))  # Remove duplicates

    def extract_phones(self, df: pd.DataFrame) -> List[str]:
        """Extract all phone numbers from messages"""
        phones = []
        phone_pattern = r'\+?\d{10,}'
        for text in df['text']:
            found = re.findall(phone_pattern, str(text))
            phones.extend(found)
        return list(set(phones))

    def extract_emails(self, df: pd.DataFrame) -> List[str]:
        """Extract all email addresses from messages"""
        emails = []
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for text in df['text']:
            found = re.findall(email_pattern, str(text))
            emails.extend(found)
        return list(set(emails))

    def analyze_chat_summary(self, df: pd.DataFrame) -> Dict:
        prompt = f"""
Chat summary request for forensic investigation.
Stats:
- Messages: {len(df)}
- Participants: {df['from'].nunique()}
- URLs shared: {df['has_url'].sum()}
- Phone numbers mentioned: {df['has_phone'].sum()}

Sample messages:
{chr(10).join(df.head(20)['text'].tolist())}

Return JSON:
{{
  "overall_risk": "LOW/MEDIUM/HIGH/CRITICAL",
  "confidence": 0-1,
  "key_findings": ["finding1", "finding2"],
  "summary": "detailed summary",
  "suspicious_indicators": ["indicator1", "indicator2"]
}}
"""
        try:
            resp = self.model.generate_content(prompt)
            return json.loads(resp.text.strip().replace("```json", "").replace("```", ""))
        except:
            return {"overall_risk": "MEDIUM", "confidence": 0.5, "summary": "Parsing failed", "key_findings": [], "suspicious_indicators": []}

    def analyze_obfuscation_table(self, df: pd.DataFrame) -> List[Dict]:
        sample = "\n".join(df['text'].tolist()[:50])
        prompt = f"""
Analyze the following chat and extract potential obfuscations (emoji, slang, codewords).
For each suspicious symbol/text, return:
- symbol: the emoji or word
- type: emoji/slang/codeword
- meaning: possible meaning
- context: 1 short sentence explaining its role in this chat

Chat sample:
{sample}

Return JSON list:
[
  {{"symbol": "", "type": "", "meaning": "", "context": ""}}
]
"""
        try:
            resp = self.model.generate_content(prompt)
            return json.loads(resp.text.strip().replace("```json","").replace("```",""))
        except:
            return []

    def extract_entities(self, df: pd.DataFrame) -> Dict:
        sample = "\n".join(df['text'].tolist()[:100])
        prompt = f"""
Extract forensically relevant entities from this chat. Return JSON:
{{
  "names": ["name1", "name2"],
  "locations": ["location1", "location2"],
  "organizations": ["org1", "org2"],
  "suspicious_terms": ["term1", "term2"]
}}

Chat:
{sample}
"""
        try:
            resp = self.model.generate_content(prompt)
            return json.loads(resp.text.strip().replace("```json", "").replace("```", ""))
        except:
            return {"names": [], "locations": [], "organizations": [], "suspicious_terms": []}

# Enhanced Visualization Functions
def create_timeline_chart(df: pd.DataFrame):
    daily = df.groupby("day").size().reset_index(name="count")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["day"],
        y=daily["count"],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#2563eb', width=3),
        fillcolor='rgba(37, 99, 235, 0.2)'
    ))
    fig.update_layout(
        title="Message activity timeline",
        xaxis_title="Date",
        yaxis_title="Message Count",
        plot_bgcolor='#ffffff',
        height=400
    )
    return fig

def create_participant_chart(df: pd.DataFrame):
    counts = df["from"].value_counts().head(10)
    fig = go.Figure(data=[go.Bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        marker=dict(color='#1e3a8a'),
        text=counts.values,
        textposition='outside'
    )])
    fig.update_layout(
        title="Top list of active participants",
        xaxis_title="Message Count",
        yaxis_title="Participant",
        plot_bgcolor='#ffffff',
        height=400
    )
    return fig

def create_temporal_heatmap(df: pd.DataFrame):
    """Hour-of-day vs Day-of-week heatmap - excellent for pattern detection"""
    df['hour'] = df['date'].dt.hour
    df['day_name'] = df['date'].dt.day_name()
    
    heatmap_data = df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        text=heatmap_data.values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Messages")
    ))
    
    fig.update_layout(
        title="Activity Heatmap: Day vs Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400
    )
    return fig

def create_message_velocity_chart(df: pd.DataFrame):
    """Detects bursts of activity - potential coordination events"""
    df['date_hour'] = df['date'].dt.floor('H')
    hourly = df.groupby('date_hour').size().reset_index(name='count')
    hourly['velocity'] = hourly['count'].rolling(window=3, center=True).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly['date_hour'],
        y=hourly['count'],
        mode='lines',
        name='Messages/Hour',
        line=dict(color='#94a3b8', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=hourly['date_hour'],
        y=hourly['velocity'],
        mode='lines',
        name='Velocity (3h avg)',
        line=dict(color='#415a77', width=3)
    ))
    
    fig.update_layout(
        title="Message velocity analysis",
        xaxis_title="Time",
        yaxis_title="Messages per Hour",
        plot_bgcolor='#ffffff',
        height=400
    )
    return fig

def create_response_time_analysis(df: pd.DataFrame):
    """Analyzes response patterns - can detect bots or coordinated behavior"""
    df_clean = df[df['time_since_prev'].notna() & (df['time_since_prev'] < 60)]  # Filter to responses within 1 hour
    
    if len(df_clean) == 0:
        return None
    
    top_users = df['from'].value_counts().head(5).index
    
    fig = go.Figure()
    for user in top_users:
        user_data = df_clean[df_clean['from'] == user]['time_since_prev']
        if len(user_data) > 0:
            fig.add_trace(go.Box(
                y=user_data,
                name=user,
                boxmean='sd'
            ))
    
    fig.update_layout(
        title="Response Time Distribution (Minutes) - Bot Detection",
        yaxis_title="Minutes Since Previous Message",
        plot_bgcolor='#ffffff',
        height=400
    )
    return fig

def create_communication_network(df: pd.DataFrame):
    """Shows who replies to whom - identifies key players"""
    top_users = df['from'].value_counts().head(8).index.tolist()
    df_filtered = df[df['from'].isin(top_users)]
    
    user_counts = df_filtered['from'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=user_counts.index,
        values=user_counts.values,
        hole=0.4,
        marker=dict(colors=px.colors.sequential.Blues_r)
    )])
    
    fig.update_layout(
        title="Communication Network: Message distribution",
        height=450
    )
    return fig

def create_content_analysis_chart(df: pd.DataFrame):
    """Analyzes message content characteristics"""
    metrics = {
        'Media Files': df['text'].str.contains(r'\.(jpg|png|gif|mp4|pdf)', regex=True, na=False).sum(),
        'Long Messages (>500 chars)': (df['msg_length'] > 500).sum(),
        'Short Messages (<10 chars)': (df['msg_length'] < 10).sum(),
    }
    
    fig = go.Figure(data=[go.Bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        marker=dict(color=['#2563eb', '#1e3a8a', '#475569']),
        text=list(metrics.values()),
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Message content characteristics",
        yaxis_title="Occurrences",
        plot_bgcolor='#ffffff',
        height=400
    )
    return fig

def create_message_length_distribution(df: pd.DataFrame):
    """Unusual message lengths can indicate automated/scripted messages"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['msg_length'],
        nbinsx=50,
        marker=dict(color='#2563eb'),
        name='Message Length'
    ))
    
    fig.update_layout(
        title="Message length distribution",
        xaxis_title="Characters",
        yaxis_title="Frequency",
        plot_bgcolor='#ffffff',
        height=400
    )
    return fig

# Helper Functions
def load_json(file):
    try:
        return pd.DataFrame(json.load(file).get("messages", []))
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

def render_metric_card(value, label, color):
    st.markdown(f"""
    <div class="metric-container" style="border-left: 4px solid {color};">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="professional-header">
        <h1 class="header-title"><i class="fas fa-search" style="margin-right: 0.5rem; color: #353535;"></i>Telegram Forensic Analyzer</h1>
        <p class="header-subtitle">AI-powered Telegram forensic analysis with forensic graphs, automated summaries, risk scoring, and obfuscation detection.</p>
    </div>
    """, unsafe_allow_html=True)

    # File Upload
    uploaded = st.file_uploader("Upload Telegram JSON Export", type="json")
    if not uploaded:
        st.info("Please upload a Telegram chat export (JSON).")
        st.markdown("""
        <div class="info-box">
            <i class="fas fa-info-circle" style="margin-right: 0.5rem; color: #2563eb;"></i>
            <strong>How to export Telegram chats:</strong><br>
            1. Open Telegram Desktop<br>
            2. Navigate to the chat you want to export, then click the three dots (⋯) in the top-right corner → Export chat history<br>
            3. Choose JSON as the export format<br>
            4. Once the export is complete, upload the downloaded JSON file here.
        </div>
        """, unsafe_allow_html=True)
        return

    # Load & process
    df = load_json(uploaded)
    if df.empty:
        st.error("Unable to load data.")
        return

    with st.spinner("Initializing Analyzer, getting your report ready..."):
        analyzer = TelegramForensicAnalyzer(API_KEY)
    st.success(f"Analysis Ready • AI Model used for analyzation: {analyzer.model_name}")

    df = analyzer.preprocess(df)

    # Metrics
    st.markdown('<div class="section-header"><h2 class="section-title"><i class="fa-solid fa-comments" style="margin-right: 0.5rem; color: #2563eb;"></i>Overview Metrics</h2></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: render_metric_card(f"{len(df):,}", "Total Messages", "#2563eb")
    with col2: render_metric_card(f"{df['from'].nunique()}", "Participants", "#1e3a8a")
    with col3: render_metric_card(f"{df['has_url'].sum()}", "URLs Shared", "#ea580c")

    # Enhanced Visualizations
    st.markdown('<div class="section-header"><h2 class="section-title"><i class="fa-solid fa-chart-line" style="margin-right: 0.5rem; color: #2563eb;"></i>Activity Analysis</h2></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Participants", "Temporal Patterns", "Velocity"])
    
    with tab1:
        st.plotly_chart(create_timeline_chart(df), use_container_width=True)
        st.plotly_chart(create_message_length_distribution(df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_participant_chart(df), use_container_width=True)
        st.plotly_chart(create_communication_network(df), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_temporal_heatmap(df), use_container_width=True)
        st.markdown("""
        <div class="warning-box">
            <i class="fas fa-exclamation-triangle" style="margin-right: 0.5rem; color: #f59e0b;"></i>
            <strong>Forensic Insight:</strong> Unusual patterns (e.g., activity only during night hours, 
            consistent hourly patterns) may indicate coordinated behavior.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.plotly_chart(create_message_velocity_chart(df), use_container_width=True)
        response_fig = create_response_time_analysis(df)
        if response_fig:
            st.plotly_chart(response_fig, use_container_width=True)

    # Content Analysis
    st.markdown('<div class="section-header"><h2 class="section-title"><i class="fas fa-chart-bar" style="margin-right: 0.5rem; color: #2563eb;"></i>Content Analysis</h2></div>', unsafe_allow_html=True)
    st.plotly_chart(create_content_analysis_chart(df), use_container_width=True)

    # AI Analysis Section
    st.markdown('<div class="section-header"><h2 class="section-title"><i class="fa-solid fa-robot" style="margin-right: 0.5rem; color: #2563eb;"></i>AI-Powered Analysis</h2></div>', unsafe_allow_html=True)
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Summary", "Obfuscation", "Entity Extraction"])
    
    with analysis_tab1:
        with st.spinner("Analyzing chat..."):
            summary = analyzer.analyze_chat_summary(df)
        
        if summary:
            risk_level = summary.get("overall_risk", "MEDIUM")
            confidence = summary.get("confidence", 0.5)
            risk_class = f"risk-{risk_level.lower()}"
            
            st.markdown(f"""
            <div class="summary-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                    <span class="risk-indicator {risk_class}">{risk_level} RISK</span>
                    <span style="font-weight:600;color:#111827">Confidence: {int(confidence*100)}%</span>
                </div>
                <div class="confidence-bar"><div class="confidence-fill" style="width:{confidence*100}%"></div></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Key Findings")
            for finding in summary.get("key_findings", []):
                st.markdown(f"• {finding}")
            
            if summary.get("suspicious_indicators"):
                st.markdown("#### Suspicious Indicators")
                for indicator in summary.get("suspicious_indicators", []):
                    st.markdown(f"• {indicator}")
            
            st.markdown("#### Summary")
            st.markdown(summary.get("summary", "No summary available."))
    
    with analysis_tab2:
        with st.spinner("Detecting obfuscation patterns..."):
            obf_table = analyzer.analyze_obfuscation_table(df)
        
        if obf_table:
            st.markdown("### Detected obfuscation patterns")
            st.table(pd.DataFrame(obf_table))
        else:
            st.info("No potential obfuscations detected.")
    
    with analysis_tab3:
        with st.spinner("Extracting forensic entities..."):
            entities = analyzer.extract_entities(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Names mentioned")
            if entities.get('names'):
                for name in entities['names'][:10]:
                    st.markdown(f"• {name}")
            else:
                st.info("No names extracted")
            
            st.markdown("#### Locations")
            if entities.get('locations'):
                for loc in entities['locations'][:10]:
                    st.markdown(f"• {loc}")
            else:
                st.info("No locations extracted")
            
            st.markdown("#### Organizations")
            if entities.get('organizations'):
                for org in entities['organizations'][:10]:
                    st.markdown(f"• {org}")
            else:
                st.info("No organizations extracted")
        
        with col2:
            st.markdown("#### Suspicious terms")
            if entities.get('suspicious_terms'):
                for term in entities['suspicious_terms'][:10]:
                    st.markdown(f"• {term}")
            else:
                st.info("No suspicious terms found")
            
            # Personal Identification Data
            st.markdown("#### Personal Identification data")
            
            # Extract URLs
            urls = analyzer.extract_urls(df)
            if urls:
                with st.expander(f"URLs found ({len(urls)})", expanded=False):
                    for url in urls[:20]:  # Show first 20
                        st.code(url, language=None)
            else:
                st.info("No URLs found")
            
            # Extract Phone Numbers
            phones = analyzer.extract_phones(df)
            if phones:
                with st.expander(f"Phone Numbers ({len(phones)})", expanded=False):
                    for phone in phones[:20]:
                        st.code(phone, language=None)
            else:
                st.info("No phone numbers found")
            
            # Extract Emails
            emails = analyzer.extract_emails(df)
            if emails:
                with st.expander(f"Email Addresses ({len(emails)})", expanded=False):
                    for email in emails[:20]:
                        st.code(email, language=None)
            else:
                st.info("No email addresses found")

if __name__ == "__main__":
    main()