import streamlit as st
import google.generativeai as genai
import uuid
from datetime import datetime, timedelta
import time
import random
import pandas as pd
import numpy as np

# --- Configuration & Initialization ---
st.set_page_config(layout="wide", page_title="Intelligent Call Campaign System")

# Constants for Simulation
AGENT_SKILLS = ["General Sales", "High-Value Sales", "Retention", "Lead Nurturing"]
LEAD_SOURCES = ["Website Signup", "Webinar Attendee", "Past Purchase", "Cold List"]
CAMPAIGN_GOALS = ["Lead Generation", "Sales Conversion", "Customer Retention", "Product Awareness"]
CALL_OUTCOMES = {
    "No Answer": 0.3,
    "Not Interested": 0.25,
    "Follow-up Required": 0.2,
    "Interested - No Sale Yet": 0.15,
    "Sale Made": 0.1 # Base probability, will be modified
}

# Session State Initialization
if 'campaign_sys_initialized' not in st.session_state:
    st.session_state.gemini_api_key_campaign = "" # Will be replaced by st.secrets if deployed
    st.session_state.campaigns = []
    st.session_state.leads = [] # List of dictionaries for leads
    st.session_state.agents = [] # List of dictionaries for agents
    st.session_state.call_log = [] # Log of all simulated calls
    st.session_state.campaign_performance = {} # campaign_id -> {metrics}
    st.session_state.simulation_time_campaign = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    st.session_state.simulation_running_campaign = False
    st.session_state.next_lead_id = 1
    st.session_state.next_agent_id_campaign = 1
    st.session_state.next_campaign_id = 1
    st.session_state.system_log_campaign = []
    st.session_state.campaign_sys_initialized = True
    st.session_state.ai_recommendations = ""

# --- Helper & Core Logic Functions ---

def log_event_campaign(message):
    timestamp = st.session_state.simulation_time_campaign.strftime("%Y-%m-%d %H:%M")
    entry = f"[{timestamp}] {message}"
    st.session_state.system_log_campaign.append(entry)
    if len(st.session_state.system_log_campaign) > 30:
        st.session_state.system_log_campaign.pop(0)

def get_gemini_campaign_response(prompt_text, temperature=0.6, max_tokens=500):
    """Generates response from Gemini, using secrets for API key if available."""
    try:
        # Try to get API key from Streamlit secrets first
        api_key_to_use = st.secrets.get("gemini_api") if hasattr(st, 'secrets') and "gemini_api" in st.secrets else st.session_state.gemini_api_key_campaign

        if not api_key_to_use:
            st.error("Gemini API Key not configured. Please set it in Streamlit secrets as 'gemini_api' or enter it manually.")
            return None

        genai.configure(api_key=api_key_to_use)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        # Prepend a role for better context
        full_prompt = (
            "You are an AI expert in call campaign management and optimization. "
            "Analyze the provided campaign data and offer actionable insights and recommendations. "
            "Focus on improving customer engagement, profit, and revenue.\n\n"
            f"{prompt_text}"
        )
        response = model.generate_content(full_prompt, generation_config=generation_config)

        if response.parts:
            return response.text
        else:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            st.warning(f"Gemini AI: No content or blocked. Reason: {block_reason}")
            log_event_campaign(f"Gemini AI: No content or blocked. Reason: {block_reason}")
            return None
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        log_event_campaign(f"Gemini API Error: {e}")
        return None

def generate_random_lead():
    lead_id = f"L{st.session_state.next_lead_id:04d}"
    st.session_state.next_lead_id += 1
    base_score = random.randint(30, 90) # General propensity score
    source = random.choice(LEAD_SOURCES)
    if source == "Past Purchase": base_score += 10
    if source == "Webinar Attendee": base_score += 5
    
    lead = {
        "id": lead_id,
        "name": f"Lead {lead_id}",
        "source": source,
        "profile_score": min(100, base_score + random.randint(-5, 5)), # 0-100
        "last_contact_date": None,
        "status": "New", # New, Contacted, Nurturing, Converted, Lost
        "assigned_campaign_id": None,
        "priority_score": 0, # Will be calculated by agentic AI
        "potential_value": random.randint(50, 1000) # Potential revenue if converted
    }
    return lead

def add_agent_campaign(name, skills):
    agent_id = f"A{st.session_state.next_agent_id_campaign:03d}"
    st.session_state.next_agent_id_campaign += 1
    st.session_state.agents.append({
        "id": agent_id, "name": name, "skills": skills,
        "status": "Available", # Available, On Call, Wrap-up
        "current_call_id": None,
        "calls_made_today": 0,
        "successful_conversions": 0,
    })
    log_event_campaign(f"Agent {name} ({skills}) added.")

def create_campaign(name, goal, target_segment_desc, offer_details, target_leads_count=50):
    campaign_id = f"C{st.session_state.next_campaign_id:03d}"
    st.session_state.next_campaign_id += 1
    st.session_state.campaigns.append({
        "id": campaign_id, "name": name, "goal": goal,
        "target_segment_description": target_segment_desc,
        "offer_details": offer_details,
        "status": "Planning", # Planning, Active, Paused, Completed
        "start_date": None, "end_date": None,
        "assigned_leads_count": 0,
        "target_leads_count": target_leads_count
    })
    st.session_state.campaign_performance[campaign_id] = {
        "calls_made": 0, "successful_conversions": 0, "total_revenue": 0,
        "engagement_score_sum": 0, "leads_contacted": 0
    }
    log_event_campaign(f"Campaign '{name}' created with ID {campaign_id}.")
    return campaign_id

def assign_leads_to_campaign(campaign_id, num_leads_to_assign):
    """Assigns best available leads to a campaign based on simple criteria."""
    campaign = next((c for c in st.session_state.campaigns if c["id"] == campaign_id), None)
    if not campaign: return 0

    # Simple targeting: prioritize higher profile_score leads not yet in a campaign
    available_leads = [l for l in st.session_state.leads if l["status"] == "New" and not l["assigned_campaign_id"]]
    available_leads.sort(key=lambda x: x["profile_score"], reverse=True)

    assigned_count = 0
    for lead in available_leads[:num_leads_to_assign]:
        lead["assigned_campaign_id"] = campaign_id
        lead["status"] = "Queued for Campaign"
        # Agentic AI: Calculate initial priority score
        lead["priority_score"] = calculate_lead_priority(lead, campaign)
        campaign["assigned_leads_count"] +=1
        assigned_count += 1
    log_event_campaign(f"Assigned {assigned_count} leads to campaign {campaign['name']}.")
    return assigned_count

def calculate_lead_priority(lead, campaign):
    """Agentic AI: Calculates a priority score for a lead within a campaign."""
    score = lead["profile_score"]
    # Adjust based on campaign goal (simplified)
    if campaign["goal"] == "Sales Conversion" and lead["source"] == "Past Purchase":
        score += 20
    elif campaign["goal"] == "Lead Generation" and lead["source"] == "Website Signup":
        score += 10
    # Add other factors like recency, specific keywords in (simulated) lead details, etc.
    return min(150, score) # Cap score

def get_next_best_lead_for_agent(agent, campaign_id):
    """Agentic AI: Finds the best lead for an agent from a specific campaign."""
    campaign_leads = [
        l for l in st.session_state.leads
        if l["assigned_campaign_id"] == campaign_id and l["status"] == "Queued for Campaign"
    ]
    if not campaign_leads:
        return None

    # Sort by priority score (higher is better)
    campaign_leads.sort(key=lambda x: x["priority_score"], reverse=True)

    # Basic skill matching (can be expanded)
    best_lead = None
    for lead in campaign_leads:
        # Simple: if agent has a relevant skill for high value or retention, prefer those leads
        if "High-Value Sales" in agent["skills"] and lead["potential_value"] > 500:
            best_lead = lead
            break
        if "Retention" in agent["skills"] and lead["source"] == "Past Purchase": # Assuming retention campaigns target past purchasers
             best_lead = lead
             break
    
    return best_lead if best_lead else campaign_leads[0] # Default to highest priority if no specific skill match

def simulate_call_attempt(agent, lead, campaign):
    """Simulates a call and its outcome."""
    agent["status"] = "On Call"
    agent["current_call_id"] = lead["id"]
    lead["status"] = "Contact Attempted"
    lead["last_contact_date"] = st.session_state.simulation_time_campaign

    # Modify base outcome probabilities
    current_outcomes = CALL_OUTCOMES.copy()
    # Higher profile score & priority increases chance of positive outcome
    positive_boost = (lead["profile_score"] / 500) + (lead["priority_score"] / 750) # Max ~0.2 + ~0.2 = 0.4
    
    current_outcomes["Sale Made"] += positive_boost * 0.5 # Sale is harder
    current_outcomes["Interested - No Sale Yet"] += positive_boost * 0.7
    current_outcomes["Follow-up Required"] += positive_boost * 0.3
    current_outcomes["Not Interested"] -= positive_boost * 0.6
    current_outcomes["No Answer"] -= positive_boost * 0.2

    # Normalize probabilities
    for k in current_outcomes: current_outcomes[k] = max(0.01, current_outcomes[k]) # Ensure no negative/zero prob
    total_prob = sum(current_outcomes.values())
    normalized_outcomes = {k: v / total_prob for k, v in current_outcomes.items()}

    outcomes, probabilities = zip(*normalized_outcomes.items())
    call_result = np.random.choice(outcomes, p=probabilities)

    # Update stats
    st.session_state.campaign_performance[campaign["id"]]["calls_made"] += 1
    agent["calls_made_today"] += 1
    revenue_earned = 0
    engagement_points = 0

    if call_result == "Sale Made":
        lead["status"] = "Converted"
        revenue_earned = lead["potential_value"] * random.uniform(0.8, 1.2) # Actual sale value
        st.session_state.campaign_performance[campaign["id"]]["successful_conversions"] += 1
        st.session_state.campaign_performance[campaign["id"]]["total_revenue"] += revenue_earned
        agent["successful_conversions"] += 1
        engagement_points = 100
    elif call_result == "Interested - No Sale Yet":
        lead["status"] = "Nurturing"
        engagement_points = 70
    elif call_result == "Follow-up Required":
        lead["status"] = "Nurturing" # Or a specific "Follow-up" status
        engagement_points = 50
    elif call_result == "Not Interested":
        lead["status"] = "Lost"
        engagement_points = 10
    else: # No Answer
        lead["status"] = "Queued for Campaign" # Re-queue for another attempt (or handle differently)
        engagement_points = 5
    
    if lead["status"] != "Queued for Campaign": # If not re-queued
         st.session_state.campaign_performance[campaign["id"]]["leads_contacted"] += 1

    st.session_state.campaign_performance[campaign["id"]]["engagement_score_sum"] += engagement_points

    st.session_state.call_log.append({
        "timestamp": st.session_state.simulation_time_campaign,
        "agent_id": agent["id"], "agent_name": agent["name"],
        "lead_id": lead["id"], "lead_name": lead["name"],
        "campaign_id": campaign["id"], "campaign_name": campaign["name"],
        "outcome": call_result, "revenue": revenue_earned, "engagement": engagement_points
    })
    log_event_campaign(f"Agent {agent['name']} called Lead {lead['name']} for Campaign '{campaign['name']}'. Outcome: {call_result}. Revenue: ${revenue_earned:.2f}")

    # Simulate wrap-up time
    # For simplicity, agent becomes available in the next step or after a fixed time
    agent["status"] = "Available" # Simplified: becomes available immediately for next sim step
    agent["current_call_id"] = None


def run_simulation_step_campaign():
    st.session_state.simulation_time_campaign += timedelta(minutes=15) # Each step is 15 mins
    log_event_campaign(f"--- Simulation Step: Time {st.session_state.simulation_time_campaign.strftime('%H:%M')} ---")

    active_campaigns = [c for c in st.session_state.campaigns if c["status"] == "Active"]
    available_agents = [a for a in st.session_state.agents if a["status"] == "Available"]

    if not active_campaigns:
        log_event_campaign("No active campaigns to process.")
        return
    if not available_agents:
        log_event_campaign("No agents available to make calls.")
        return

    # Assign calls fairly or based on campaign priority
    for agent in available_agents:
        # Simple: pick the first active campaign, or could be more complex
        # For now, let agents pick from any active campaign, or assign them dedicated campaigns
        # This round-robin attempt across campaigns for each agent
        for campaign in active_campaigns:
            if agent["status"] != "Available": break # Agent took a call

            next_lead = get_next_best_lead_for_agent(agent, campaign["id"])
            if next_lead:
                simulate_call_attempt(agent, next_lead, campaign)
                break # Agent makes one call per step for simplicity
    
    # Check for campaign completion (simplified)
    for campaign in active_campaigns:
        perf = st.session_state.campaign_performance[campaign["id"]]
        if perf["calls_made"] >= campaign["target_leads_count"] * 1.5 : # e.g., target 50 leads, allow 75 calls
            # campaign["status"] = "Completed" # Or based on actual leads processed.
            # For this example, let's assume target leads count means distinct leads to try to contact
            if perf["leads_contacted"] >= campaign["target_leads_count"]:
                 campaign["status"] = "Completed"
                 log_event_campaign(f"Campaign {campaign['name']} marked as completed (target leads contacted).")


# --- Streamlit UI ---

# Sidebar
with st.sidebar:
    st.header("⚙️ System Setup & Controls")
    # API Key Input (primarily for local testing if secrets not available)
    if not (hasattr(st, 'secrets') and "gemini_api" in st.secrets):
        st.session_state.gemini_api_key_campaign = st.text_input(
            "Google AI Studio API Key (Campaigns)",
            type="password",
            value=st.session_state.gemini_api_key_campaign,
            key="gemini_api_key_c",
            help="Enter if 'gemini_api' not in Streamlit secrets."
        )

    st.subheader("🚀 Manage Campaigns")
    with st.expander("Create New Campaign", expanded=False):
        with st.form("new_campaign_form", clear_on_submit=True):
            c_name = st.text_input("Campaign Name", f"Q{datetime.now().month} Sales Drive")
            c_goal = st.selectbox("Campaign Goal", CAMPAIGN_GOALS)
            c_target_desc = st.text_area("Target Segment Description", "Customers interested in new tech products, high engagement score.")
            c_offer = st.text_area("Offer Details", "20% off new XYZ gadget, free shipping.")
            c_num_leads = st.number_input("Target Number of Leads to Contact", 10, 1000, 50)
            submitted_campaign = st.form_submit_button("Create Campaign")
            if submitted_campaign and c_name:
                create_campaign(c_name, c_goal, c_target_desc, c_offer, c_num_leads)
                st.success(f"Campaign '{c_name}' created.")

    if st.session_state.campaigns:
        selected_campaign_id_action = st.selectbox(
            "Select Campaign for Actions",
            options=[c["id"] for c in st.session_state.campaigns],
            format_func=lambda x: next(c["name"] for c in st.session_state.campaigns if c["id"] == x),
            key="sel_camp_action_sb"
        )
        campaign_to_act_on = next((c for c in st.session_state.campaigns if c["id"] == selected_campaign_id_action), None)
        if campaign_to_act_on:
            if campaign_to_act_on["status"] == "Planning":
                if st.button(f"🚀 Activate Campaign: {campaign_to_act_on['name']}", key=f"act_{selected_campaign_id_action}"):
                    num_assigned = assign_leads_to_campaign(selected_campaign_id_action, campaign_to_act_on["target_leads_count"])
                    if num_assigned > 0:
                        campaign_to_act_on["status"] = "Active"
                        campaign_to_act_on["start_date"] = st.session_state.simulation_time_campaign
                        log_event_campaign(f"Campaign {campaign_to_act_on['name']} activated.")
                        st.rerun()
                    else:
                        st.warning("No new leads available to assign for activation.")

            elif campaign_to_act_on["status"] == "Active":
                if st.button(f"⏸️ Pause Campaign: {campaign_to_act_on['name']}", key=f"pause_{selected_campaign_id_action}"):
                    campaign_to_act_on["status"] = "Paused"
                    log_event_campaign(f"Campaign {campaign_to_act_on['name']} paused.")
                    st.rerun()
            elif campaign_to_act_on["status"] == "Paused":
                 if st.button(f"▶️ Resume Campaign: {campaign_to_act_on['name']}", key=f"resume_{selected_campaign_id_action}"):
                    campaign_to_act_on["status"] = "Active"
                    log_event_campaign(f"Campaign {campaign_to_act_on['name']} resumed.")
                    st.rerun()


    st.subheader("👥 Manage Agents")
    with st.expander("Add New Agent", expanded=False):
        with st.form("new_agent_form_campaign", clear_on_submit=True):
            a_name = st.text_input("Agent Name", f"Agent {st.session_state.next_agent_id_campaign}")
            a_skills = st.multiselect("Agent Skills", AGENT_SKILLS, default=[AGENT_SKILLS[0]])
            submitted_agent = st.form_submit_button("Add Agent")
            if submitted_agent and a_name and a_skills:
                add_agent_campaign(a_name, a_skills)
                st.success(f"Agent '{a_name}' added.")

    st.subheader("💧 Manage Leads")
    num_leads_to_gen = st.number_input("Generate Random Leads", 0, 100, 10, key="gen_leads_sb")
    if st.button("Generate Leads", key="gen_leads_btn_sb"):
        for _ in range(num_leads_to_gen):
            st.session_state.leads.append(generate_random_lead())
        log_event_campaign(f"Generated {num_leads_to_gen} new random leads.")
        st.rerun()


    st.subheader("⚙️ Simulation Control")
    if st.session_state.simulation_running_campaign:
        if st.button("⏹️ Pause Simulation", key="pause_sim_c"):
            st.session_state.simulation_running_campaign = False
            log_event_campaign("Simulation Paused.")
            st.rerun()
    else:
        if st.button("▶️ Run Simulation", key="run_sim_c"):
            if not st.session_state.agents: st.warning("Add agents to run simulation."); st.stop()
            if not any(c["status"]=="Active" for c in st.session_state.campaigns): st.warning("Activate a campaign to run simulation."); st.stop()
            st.session_state.simulation_running_campaign = True
            log_event_campaign("Simulation Started/Resumed.")
            st.rerun()

    if not st.session_state.simulation_running_campaign:
            if st.button("⏭️ Simulate Next Step", key="next_step_c"):
                if not st.session_state.agents: st.warning("Add agents to run simulation."); st.stop()
                if not any(c["status"]=="Active" for c in st.session_state.campaigns): st.warning("Activate a campaign to run simulation."); st.stop()
                run_simulation_step_campaign()
                st.rerun()

    simulation_speed_campaign = st.slider("Sim Speed (steps/sec)", 0.1, 2.0, 0.5, 0.1, key="sim_speed_c", disabled=not st.session_state.simulation_running_campaign)

    st.subheader("📜 System Log")
    log_container_c = st.container(height=200)
    for log_c in reversed(st.session_state.system_log_campaign):
        log_container_c.caption(log_c)


# Main Application Area
st.title("📞 Intelligent Call Campaign Management System")
st.markdown(f"**Simulation Time: {st.session_state.simulation_time_campaign.strftime('%A, %B %d, %Y %H:%M')}**")

tab_overview, tab_campaigns, tab_leads, tab_agents, tab_ai_insights = st.tabs([
    "📊 Overview", "🚀 Campaigns", "💧 Leads", "👥 Agents", "💡 AI Optimizer"
])

with tab_overview:
    st.header("Overall Performance Snapshot")
    if not st.session_state.call_log:
        st.info("No calls made yet. Run the simulation to see data.")
    else:
        total_calls = sum(perf["calls_made"] for perf in st.session_state.campaign_performance.values())
        total_conversions = sum(perf["successful_conversions"] for perf in st.session_state.campaign_performance.values())
        total_revenue_all = sum(perf["total_revenue"] for perf in st.session_state.campaign_performance.values())
        
        avg_engagement = 0
        total_engagement_sum = sum(perf["engagement_score_sum"] for perf in st.session_state.campaign_performance.values())
        total_leads_contacted_all = sum(perf["leads_contacted"] for perf in st.session_state.campaign_performance.values())
        if total_leads_contacted_all > 0:
            avg_engagement = total_engagement_sum / total_leads_contacted_all


        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Calls Made", total_calls)
        col2.metric("Total Conversions", total_conversions)
        col3.metric("Total Revenue", f"${total_revenue_all:,.2f}")
        col4.metric("Avg. Engagement/Contacted Lead", f"{avg_engagement:.1f}")

        st.subheader("Recent Call Log")
        call_df = pd.DataFrame(st.session_state.call_log)
        if not call_df.empty:
            st.dataframe(call_df.tail(10).iloc[::-1], use_container_width=True) # Show last 10, newest first

with tab_campaigns:
    st.header("Campaign Details & Performance")
    if not st.session_state.campaigns:
        st.info("No campaigns created yet. Create one from the sidebar.")
    else:
        campaign_data_display = []
        for camp in st.session_state.campaigns:
            perf = st.session_state.campaign_performance.get(camp["id"], {})
            avg_camp_engagement = (perf.get("engagement_score_sum",0) / perf.get("leads_contacted",1)) if perf.get("leads_contacted",0) > 0 else 0
            conversion_rate = (perf.get("successful_conversions",0) / perf.get("leads_contacted",1)) * 100 if perf.get("leads_contacted",0) > 0 else 0

            campaign_data_display.append({
                "ID": camp["id"], "Name": camp["name"], "Goal": camp["goal"], "Status": camp["status"],
                "Target Leads": camp["target_leads_count"],
                "Leads Contacted": perf.get("leads_contacted", 0),
                "Calls": perf.get("calls_made", 0),
                "Conversions": perf.get("successful_conversions", 0),
                "Revenue": f"${perf.get('total_revenue', 0):,.2f}",
                "Avg. Engagement": f"{avg_camp_engagement:.1f}",
                "Conv. Rate (%)": f"{conversion_rate:.1f}%"
            })
        st.dataframe(pd.DataFrame(campaign_data_display), use_container_width=True)

with tab_leads:
    st.header("Lead Management")
    if not st.session_state.leads:
        st.info("No leads available. Generate some from the sidebar.")
    else:
        # Add filters for leads
        status_filter = st.multiselect("Filter by Status:", options=list(set(l['status'] for l in st.session_state.leads)), key="lead_status_filter")
        campaign_filter_lead = st.selectbox("Filter by Campaign:", options=["All"] + [c['id'] + " - " + c['name'] for c in st.session_state.campaigns], key="lead_camp_filter")

        filtered_leads = st.session_state.leads
        if status_filter:
            filtered_leads = [l for l in filtered_leads if l['status'] in status_filter]
        if campaign_filter_lead != "All":
            camp_id_to_filter = campaign_filter_lead.split(" - ")[0]
            filtered_leads = [l for l in filtered_leads if l['assigned_campaign_id'] == camp_id_to_filter]

        st.metric("Displaying Leads", len(filtered_leads))
        lead_df_display = pd.DataFrame(filtered_leads)[["id", "name", "source", "profile_score", "priority_score", "status", "assigned_campaign_id", "potential_value"]]
        st.dataframe(lead_df_display, use_container_width=True)


with tab_agents:
    st.header("Agent Roster & Performance")
    if not st.session_state.agents:
        st.info("No agents added yet. Add agents from the sidebar.")
    else:
        agent_data_display = []
        for agent in st.session_state.agents:
            agent_data_display.append({
                "ID": agent["id"], "Name": agent["name"], "Skills": ", ".join(agent["skills"]),
                "Status": agent["status"],
                "Current Call Lead ID": agent["current_call_id"][:5] if agent["current_call_id"] else "N/A",
                "Calls Today": agent["calls_made_today"],
                "Conversions Today": agent["successful_conversions"]
            })
        st.dataframe(pd.DataFrame(agent_data_display), use_container_width=True)

with tab_ai_insights:
    st.header("💡 Gemini AI Campaign Optimizer")
    st.markdown("Get AI-powered recommendations to enhance your campaign strategies.")

    selected_campaign_id_ai = st.selectbox(
        "Select Campaign for AI Analysis:",
        options=["All Active Campaigns"] + [c["id"] + " - " + c["name"] for c in st.session_state.campaigns if c["status"] != "Planning"],
        key="sel_camp_ai"
    )

    if st.button("🤖 Analyze and Suggest Optimizations", key="gemini_opt_btn"):
        # Prepare data for Gemini
        prompt_data = "Current Campaign System State:\n"
        prompt_data += f"- Simulation Time: {st.session_state.simulation_time_campaign.strftime('%Y-%m-%d %H:%M')}\n"
        
        campaigns_to_analyze = []
        if selected_campaign_id_ai == "All Active Campaigns":
            campaigns_to_analyze = [c for c in st.session_state.campaigns if c["status"] == "Active"]
        else:
            camp_id = selected_campaign_id_ai.split(" - ")[0]
            sel_camp = next((c for c in st.session_state.campaigns if c["id"] == camp_id), None)
            if sel_camp: campaigns_to_analyze.append(sel_camp)

        if not campaigns_to_analyze:
            st.warning("No suitable campaigns selected or active for analysis.")
        else:
            for camp_obj in campaigns_to_analyze:
                perf = st.session_state.campaign_performance.get(camp_obj["id"], {})
                prompt_data += f"\nCampaign: {camp_obj['name']} (ID: {camp_obj['id']})\n"
                prompt_data += f"  Status: {camp_obj['status']}, Goal: {camp_obj['goal']}\n"
                prompt_data += f"  Target Segment: {camp_obj['target_segment_description']}\n"
                prompt_data += f"  Offer: {camp_obj['offer_details']}\n"
                prompt_data += f"  Performance:\n"
                prompt_data += f"    Calls Made: {perf.get('calls_made',0)}\n"
                prompt_data += f"    Leads Contacted: {perf.get('leads_contacted',0)}\n"
                prompt_data += f"    Conversions: {perf.get('successful_conversions',0)}\n"
                prompt_data += f"    Total Revenue: ${perf.get('total_revenue',0):.2f}\n"
                avg_eng = (perf.get('engagement_score_sum',0) / perf.get('leads_contacted',1)) if perf.get('leads_contacted',0)>0 else 0
                prompt_data += f"    Avg. Engagement per Contacted Lead: {avg_eng:.1f}\n"

            prompt_data += "\nRecent Call Log Summary (last 5 calls):\n"
            for call in st.session_state.call_log[-5:]:
                 prompt_data += f"- Agent {call['agent_name']} called Lead {call['lead_name']} for {call['campaign_name']}: Outcome {call['outcome']}, Revenue ${call['revenue']:.2f}\n"


            final_prompt = (
                f"{prompt_data}\n\n"
                "Based on the above data, provide:\n"
                "1. A concise summary of the current campaign performance (highlight strengths and weaknesses).\n"
                "2. Three actionable recommendations to improve customer engagement, profit, AND revenue for the selected campaign(s). Be specific (e.g., 'Consider A/B testing offer X vs Y for segment Z', or 'Refine lead prioritization for campaign A to focus on leads with scores > X').\n"
                "3. If multiple campaigns analyzed, identify any cross-campaign learnings or opportunities.\n"
                "4. Suggest one specific element (e.g., script opening, offer detail, target audience refinement) that could be A/B tested next, and why."
            )

            with st.spinner("Gemini is crafting campaign optimization strategies..."):
                st.session_state.ai_recommendations = get_gemini_campaign_response(final_prompt, temperature=0.7, max_tokens=800)
            
            if st.session_state.ai_recommendations:
                st.success("AI Recommendations Ready!")
            else:
                st.error("Failed to get recommendations from Gemini.")
    
    if st.session_state.ai_recommendations:
        st.markdown("---")
        st.markdown(st.session_state.ai_recommendations)


# Auto-run simulation loop
if st.session_state.simulation_running_campaign:
    active_camps = any(c["status"] == "Active" for c in st.session_state.campaigns)
    if st.session_state.agents and active_camps:
        run_simulation_step_campaign()
        time.sleep(1.0 / simulation_speed_campaign)
        st.rerun()
    else:
        if not st.session_state.agents: log_event_campaign("Paused: No agents available.")
        if not active_camps: log_event_campaign("Paused: No active campaigns.")
        st.session_state.simulation_running_campaign = False # Auto-pause
        st.rerun()