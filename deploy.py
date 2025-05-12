import streamlit as st
import google.generativeai as genai
import uuid
from datetime import datetime, timedelta
import time
import random
import pandas as pd
import numpy as np

# --- Configuration & Initialization ---
st.set_page_config(layout="wide", page_title="Intelligent Event Campaign System")

# Constants
EVENT_TYPES = ["Webinar", "Workshop", "Conference (Virtual)", "Product Launch", "Networking Mixer"]
PROMO_CHANNELS = ["Email Blast", "Social Media Organic", "Social Media Paid", "Partner Promotion", "Website Banner"]
AUDIENCE_INTEREST_TAGS = ["Tech", "Marketing", "Sales", "Finance", "HR", "Startups", "Enterprise", "AI", "Sustainability"]

# Session State
if 'event_mgmt_sys_initialized' not in st.session_state:
    st.session_state.gemini_api_key_event = "" # For local testing, use st.secrets for deployment
    st.session_state.events = [] # List of event campaign dicts
    st.session_state.audience_pool = [] # List of potential attendee dicts
    st.session_state.event_registrations = {} # event_id -> set of audience_ids
    st.session_state.event_attendance = {} # event_id -> set of audience_ids (attended)
    st.session_state.event_performance_metrics = {} # event_id -> dict of metrics
    st.session_state.system_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    st.session_state.simulation_running_event = False
    st.session_state.next_event_id_counter = 1
    st.session_state.next_audience_id_counter = 1
    st.session_state.system_log_event = []
    st.session_state.ai_event_recommendations = ""
    st.session_state.event_mgmt_sys_initialized = True

# --- Helper Functions ---
def log_event_activity(message):
    timestamp = st.session_state.system_time.strftime("%Y-%m-%d %H:%M")
    entry = f"[{timestamp}] {message}"
    st.session_state.system_log_event.append(entry)
    if len(st.session_state.system_log_event) > 30:
        st.session_state.system_log_event.pop(0)

def get_gemini_event_response(prompt_text, temperature=0.7, max_tokens=600):
    try:
        api_key_to_use = st.secrets.get("gemini_api") if hasattr(st, 'secrets') and "gemini_api" in st.secrets else st.session_state.gemini_api_key_event
        if not api_key_to_use:
            st.error("Gemini API Key not configured. Set 'gemini_api' in Streamlit secrets or enter manually.")
            return None

        genai.configure(api_key=api_key_to_use)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        generation_config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        
        full_prompt = (
            "You are an AI expert in event campaign management and marketing. "
            "Analyze the provided event data and offer actionable insights, content ideas, and strategic recommendations. "
            "Focus on improving registrations, attendance, engagement, and overall event ROI.\n\n"
            f"{prompt_text}"
        )
        response = model.generate_content(full_prompt, generation_config=generation_config)
        return response.text if response.parts else None
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        log_event_activity(f"Gemini API Error: {e}")
        return None

def generate_random_audience_member():
    audience_id = f"AU{st.session_state.next_audience_id_counter:04d}"
    st.session_state.next_audience_id_counter += 1
    member = {
        "id": audience_id,
        "name": f"Audience {audience_id}",
        "interest_tags": random.sample(AUDIENCE_INTEREST_TAGS, k=random.randint(1, 3)),
        "engagement_score": random.randint(20, 90), # General engagement with brand/past events
        "potential_value": random.randint(100, 2000) if random.random() < 0.3 else 0, # Potential if they become a customer
        "last_promo_interaction_days_ago": random.randint(5, 90)
    }
    return member

def create_event_campaign(name, event_type, target_audience_desc, goals, content_highlights, promo_channels, budget, event_date_offset_days=30):
    event_id = f"EVT{st.session_state.next_event_id_counter:03d}"
    st.session_state.next_event_id_counter += 1
    event_date = st.session_state.system_time + timedelta(days=event_date_offset_days)
    # Define key dates relative to event_date
    promo_start_date = event_date - timedelta(days=21)
    early_bird_ends_date = event_date - timedelta(days=14)
    reminder_1_date = event_date - timedelta(days=3)
    reminder_2_date = event_date - timedelta(days=1)

    event = {
        "id": event_id, "name": name, "type": event_type, "status": "Planning",
        "target_audience_description": target_audience_desc,
        "goals": goals, # List of strings
        "content_highlights": content_highlights,
        "promo_channels": promo_channels, # List of strings
        "budget_allocated": budget,
        "event_date": event_date,
        "promo_start_date": promo_start_date,
        "early_bird_ends_date": early_bird_ends_date,
        "reminder_1_date": reminder_1_date,
        "reminder_2_date": reminder_2_date,
        "promotions_sent_count": 0,
        "reminders_sent_count": 0,
        "simulated_promo_effectiveness": random.uniform(0.01, 0.05) # Base % of targeted audience that might register per promo wave
    }
    st.session_state.events.append(event)
    st.session_state.event_registrations[event_id] = set()
    st.session_state.event_attendance[event_id] = set()
    st.session_state.event_performance_metrics[event_id] = {
        "registrations": 0, "attendance": 0, "leads_generated": 0, "cost_incurred": 0,
        "estimated_revenue_impact": 0, "promo_reach": 0
    }
    log_event_activity(f"Event Campaign '{name}' (ID: {event_id}) created for {event_date.strftime('%Y-%m-%d')}.")
    return event_id

def get_target_audience_for_event(event, audience_pool, max_target_count=500):
    """Agentic AI: Selects a target audience based on event description and audience profiles."""
    # Simple keyword matching for now
    target_audience = []
    event_keywords = set(tag.lower() for tag in event.get("target_audience_description", "").replace(",", " ").split() if tag in AUDIENCE_INTEREST_TAGS)
    if not event_keywords and event.get("type"): # Fallback to event type
        event_keywords.add(event["type"].split()[0].lower())


    for member in audience_pool:
        member_interests = set(tag.lower() for tag in member["interest_tags"])
        if event_keywords & member_interests: # If any overlap
            # Prioritize higher engagement score and recency
            score = member["engagement_score"] - member["last_promo_interaction_days_ago"]/10
            target_audience.append((score, member))
    
    target_audience.sort(key=lambda x: x[0], reverse=True)
    return [m[1] for m in target_audience[:max_target_count]]


def simulate_event_campaign_step():
    st.session_state.system_time += timedelta(days=1) # Simulate daily steps
    log_event_activity(f"--- System Day: {st.session_state.system_time.strftime('%Y-%m-%d')} ---")

    for event in st.session_state.events:
        if event["status"] in ["Completed", "Cancelled"]:
            continue

        metrics = st.session_state.event_performance_metrics[event["id"]]

        # Promotion Phase
        if event["status"] == "Active - Promotion" and st.session_state.system_time >= event["promo_start_date"] and st.session_state.system_time < event["event_date"]:
            if event["promotions_sent_count"] < 3: # Limit promo blasts
                targeted_audience = get_target_audience_for_event(event, st.session_state.audience_pool, 200 * (event["promotions_sent_count"] + 1))
                metrics["promo_reach"] = max(metrics["promo_reach"], len(targeted_audience))
                
                new_registrations_this_wave = 0
                for member in targeted_audience:
                    if member["id"] not in st.session_state.event_registrations[event["id"]]:
                        # Probability based on engagement, event appeal (effectiveness), and if early bird is active
                        reg_prob = event["simulated_promo_effectiveness"] + (member["engagement_score"] / 2000)
                        if st.session_state.system_time <= event["early_bird_ends_date"]:
                            reg_prob *= 1.5 # Early bird boost

                        if random.random() < reg_prob:
                            st.session_state.event_registrations[event["id"]].add(member["id"])
                            new_registrations_this_wave +=1
                
                metrics["registrations"] = len(st.session_state.event_registrations[event["id"]])
                metrics["cost_incurred"] += event["budget_allocated"] * 0.1 # Assume 10% budget per promo wave
                event["promotions_sent_count"] += 1
                log_event_activity(f"Event '{event['name']}': Promo wave {event['promotions_sent_count']} sent. {new_registrations_this_wave} new registrations. Total: {metrics['registrations']}")
        
        # Reminder Phase
        if event["status"] == "Active - Promotion" and metrics["registrations"] > 0:
            if st.session_state.system_time == event["reminder_1_date"] or st.session_state.system_time == event["reminder_2_date"]:
                event["reminders_sent_count"] += 1
                metrics["cost_incurred"] += event["budget_allocated"] * 0.02 # Small cost for reminders
                log_event_activity(f"Event '{event['name']}': Reminder {event['reminders_sent_count']} sent to {metrics['registrations']} registrants.")

        # Event Day - Simulate Attendance
        if st.session_state.system_time == event["event_date"]:
            event["status"] = "Event Day"
            registered_ids = st.session_state.event_registrations[event["id"]]
            attended_count = 0
            for reg_id in registered_ids:
                # Show-up probability (e.g. 50-80%)
                if random.random() < (0.65 + random.uniform(-0.15, 0.15)): # Average 65% show-up
                    st.session_state.event_attendance[event["id"]].add(reg_id)
                    attended_count +=1
            metrics["attendance"] = attended_count
            log_event_activity(f"Event '{event['name']}' is TODAY! {attended_count} out of {metrics['registrations']} attended.")
            event["status"] = "Post-Event" # Move to post-event phase for next day's processing

        # Post-Event Phase (Lead Gen, Revenue)
        if event["status"] == "Post-Event" and st.session_state.system_time > event["event_date"]:
            if metrics["leads_generated"] == 0 and metrics["attendance"] > 0: # Simulate once
                attendee_ids = st.session_state.event_attendance[event["id"]]
                leads_from_event = 0
                revenue_from_event = 0
                for aud_id in attendee_ids:
                    member = next((m for m in st.session_state.audience_pool if m["id"] == aud_id), None)
                    if member:
                        # Lead conversion probability (e.g., 10-30% of attendees become leads)
                        if random.random() < (0.2 + (member["engagement_score"] / 1000)):
                            leads_from_event += 1
                            # Revenue from converted lead
                            if member["potential_value"] > 0 and random.random() < 0.5: # 50% of leads with potential convert to value
                                revenue_from_event += member["potential_value"] * random.uniform(0.7,1.0)
                
                metrics["leads_generated"] = leads_from_event
                metrics["estimated_revenue_impact"] = revenue_from_event
                metrics["cost_incurred"] += event["budget_allocated"] * 0.05 # Post-event follow-up cost
                log_event_activity(f"Event '{event['name']}': Post-event processing. Leads: {leads_from_event}, Est. Revenue: ${revenue_from_event:,.2f}")
                event["status"] = "Completed" # Mark as fully completed after post-event actions

        # Risk Assessment (Agentic AI)
        if event["status"] == "Active - Promotion" and st.session_state.system_time < event["event_date"] - timedelta(days=7):
            days_into_promo = (st.session_state.system_time - event["promo_start_date"]).days
            expected_reg_velocity = (event["budget_allocated"] / 5000) * 10 # Arbitrary: Higher budget = higher expected daily reg
            if days_into_promo > 3 and metrics["registrations"] < (expected_reg_velocity * days_into_promo * 0.5): # Less than 50% of expected
                 log_event_activity(f"RISK: Event '{event['name']}' has low registration velocity ({metrics['registrations']} regs after {days_into_promo} promo days).")


# --- Streamlit UI ---
# Sidebar
with st.sidebar:
    st.header("⚙️ System Setup & Controls")
    if not (hasattr(st, 'secrets') and "gemini_api" in st.secrets):
        st.session_state.gemini_api_key_event = st.text_input(
            "Google AI API Key (Events)", type="password", value=st.session_state.gemini_api_key_event, key="gemini_api_key_evt"
        )

    st.subheader("🎉 Create New Event Campaign")
    with st.expander("Event Details Form", expanded=False):
        with st.form("new_event_form", clear_on_submit=True):
            evt_name = st.text_input("Event Name", f"My Awesome {random.choice(EVENT_TYPES)}")
            evt_type = st.selectbox("Event Type", EVENT_TYPES)
            evt_target_desc = st.text_area("Target Audience Description", "Professionals interested in AI and Marketing, mid-career.")
            evt_goals = st.multiselect("Event Goals", ["Lead Generation", "Brand Awareness", "Sales", "Networking"])
            evt_content = st.text_area("Key Content/Speakers", "Keynote by Dr. AI, Workshop on Prompt Engineering.")
            evt_promo_channels = st.multiselect("Promotion Channels", PROMO_CHANNELS, default=PROMO_CHANNELS[:2])
            evt_budget = st.number_input("Allocated Budget ($)", 100, 10000, 1000, 100)
            evt_date_offset = st.slider("Event Date (days from today)", 7, 90, 30)
            submitted_event = st.form_submit_button("Create Event Campaign")
            if submitted_event and evt_name:
                create_event_campaign(evt_name, evt_type, evt_target_desc, evt_goals, evt_content, evt_promo_channels, evt_budget, evt_date_offset)
                st.success(f"Event '{evt_name}' created.")

    st.subheader("👥 Manage Audience Pool")
    num_audience_to_gen = st.number_input("Generate Random Audience Members", 0, 200, 20, key="gen_aud_sb")
    if st.button("Generate Audience", key="gen_aud_btn_sb"):
        for _ in range(num_audience_to_gen):
            st.session_state.audience_pool.append(generate_random_audience_member())
        log_event_activity(f"Generated {num_audience_to_gen} new audience members.")
        st.rerun()
    st.caption(f"Current Audience Pool Size: {len(st.session_state.audience_pool)}")

    st.subheader("⚙️ Simulation Control")
    if st.session_state.simulation_running_event:
        if st.button("⏹️ Pause Simulation", key="pause_sim_evt"):
            st.session_state.simulation_running_event = False
            log_event_activity("Simulation Paused.")
            st.rerun()
    else:
        if st.button("▶️ Run Simulation", key="run_sim_evt"):
            if not st.session_state.events: st.warning("Create an event first!"); st.stop()
            if not st.session_state.audience_pool: st.warning("Generate audience members first!"); st.stop()
            st.session_state.simulation_running_event = True
            log_event_activity("Simulation Started/Resumed.")
            st.rerun()

    if not st.session_state.simulation_running_event:
            if st.button("⏭️ Simulate Next Day", key="next_day_evt"):
                if not st.session_state.events: st.warning("Create an event first!"); st.stop()
                if not st.session_state.audience_pool: st.warning("Generate audience members first!"); st.stop()
                simulate_event_campaign_step()
                st.rerun()

    simulation_speed_event = st.slider("Sim Speed (days/sec)", 0.1, 2.0, 0.5, 0.1, key="sim_speed_evt", disabled=not st.session_state.simulation_running_event)

    st.subheader("📜 System Activity Log")
    log_container_evt = st.container(height=200)
    for log_e in reversed(st.session_state.system_log_event):
        log_container_evt.caption(log_e)

# Main Application Area
st.title("🎉 Intelligent Event Campaign Management System")
st.markdown(f"**System Date: {st.session_state.system_time.strftime('%A, %B %d, %Y')}**")

tab_dashboard, tab_event_list, tab_audience, tab_ai_optimizer = st.tabs([
    "📊 Dashboard", "🗓️ Event Campaigns", "👥 Audience Insights", "💡 AI Optimizer"
])

with tab_dashboard:
    st.header("Overall Event Ecosystem Snapshot")
    if not st.session_state.events:
        st.info("No event campaigns created yet.")
    else:
        active_events = len([e for e in st.session_state.events if e["status"] not in ["Completed", "Cancelled", "Planning"]])
        total_registrations_all = sum(m.get("registrations",0) for m in st.session_state.event_performance_metrics.values())
        total_attendance_all = sum(m.get("attendance",0) for m in st.session_state.event_performance_metrics.values())
        total_revenue_impact_all = sum(m.get("estimated_revenue_impact",0) for m in st.session_state.event_performance_metrics.values())
        total_cost_all = sum(m.get("cost_incurred",0) for m in st.session_state.event_performance_metrics.values())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Events Managed", len(st.session_state.events))
        col2.metric("Active Event Campaigns", active_events)
        col3.metric("Total Registrations (All Events)", f"{total_registrations_all:,}")
        col4.metric("Total Attendance (All Events)", f"{total_attendance_all:,}")
        
        col5, col6, col7, _ = st.columns(4)
        col5.metric("Total Leads Generated (All)", f"{sum(m.get('leads_generated',0) for m in st.session_state.event_performance_metrics.values()):,}")
        col6.metric("Total Est. Revenue Impact", f"${total_revenue_impact_all:,.2f}")
        col7.metric("Total Campaign Costs", f"${total_cost_all:,.2f}")

        st.subheader("Upcoming Events (Next 30 Days)")
        upcoming_events_df = pd.DataFrame([
            {"ID": e["id"], "Name": e["name"], "Type": e["type"], "Event Date": e["event_date"].strftime("%Y-%m-%d"), "Status": e["status"], "Registrations": st.session_state.event_performance_metrics[e["id"]].get("registrations",0)}
            for e in st.session_state.events
            if e["event_date"] >= st.session_state.system_time and e["event_date"] <= st.session_state.system_time + timedelta(days=30)
        ]).sort_values(by="Event Date")
        if not upcoming_events_df.empty:
            st.dataframe(upcoming_events_df, use_container_width=True)
        else:
            st.info("No events scheduled in the next 30 days.")

with tab_event_list:
    st.header("All Event Campaigns")
    if not st.session_state.events:
        st.info("No event campaigns created yet. Create one from the sidebar.")
    else:
        event_data_display = []
        for event in st.session_state.events:
            metrics = st.session_state.event_performance_metrics.get(event["id"], {})
            show_up_rate = (metrics.get("attendance", 0) / metrics.get("registrations", 1)) * 100 if metrics.get("registrations", 0) > 0 else 0
            cost_per_attendee = metrics.get("cost_incurred", 0) / metrics.get("attendance", 1) if metrics.get("attendance", 0) > 0 else 0
            
            # Action buttons
            actions = []
            if event["status"] == "Planning":
                actions.append(f"<button name='activate_event' value='{event['id']}'>🚀 Activate</button>")
            elif "Active" in event["status"]:
                 actions.append(f"<button name='pause_event' value='{event['id']}'>⏸️ Pause</button>")
            elif event["status"] == "Paused":
                 actions.append(f"<button name='resume_event' value='{event['id']}'>▶️ Resume</button>")
            # For a real app, these buttons would trigger callbacks. Here, we'd need more complex state handling or forms.
            # For simplicity, actions are illustrative. Actual activation is via status change in simulation.

            event_data_display.append({
                "ID": event["id"], "Name": event["name"], "Type": event["type"],
                "Event Date": event["event_date"].strftime("%Y-%m-%d"), "Status": event["status"],
                "Regs": metrics.get("registrations", 0),
                "Attendance": metrics.get("attendance", 0),
                "Show-up %": f"{show_up_rate:.1f}%",
                "Leads": metrics.get("leads_generated", 0),
                "Cost": f"${metrics.get('cost_incurred', 0):,.0f}",
                "CPA": f"${cost_per_attendee:,.2f}",
                "Revenue": f"${metrics.get('estimated_revenue_impact', 0):,.0f}",
                # "Actions": " ".join(actions) # Illustrative
            })
        
        df_events = pd.DataFrame(event_data_display)
        st.dataframe(df_events, use_container_width=True) # unsafe_allow_html=True for buttons

        # Logic for activating an event (simplified here - better in sidebar with forms)
        for e in st.session_state.events:
            if e["status"] == "Planning" and st.session_state.system_time >= (e["event_date"] - timedelta(days=45)): # Auto-plan activation
                if st.button(f"🚀 Activate Campaign: {e['name']}", key=f"activate_{e['id']}"):
                     e["status"] = "Active - Promotion"
                     log_event_activity(f"Event '{e['name']}' promo phase activated.")
                     st.rerun()


with tab_audience:
    st.header("Audience Pool Insights")
    if not st.session_state.audience_pool:
        st.info("Audience pool is empty. Generate some members from the sidebar.")
    else:
        st.metric("Total Audience Members in Pool", len(st.session_state.audience_pool))
        
        # Display a sample of the audience pool
        audience_df_sample = pd.DataFrame(st.session_state.audience_pool).sample(min(len(st.session_state.audience_pool), 20))
        st.dataframe(audience_df_sample[["id", "name", "interest_tags", "engagement_score", "potential_value"]], use_container_width=True)
        
        # TODO: Add charts for interest distribution, engagement scores, etc.

with tab_ai_optimizer:
    st.header("💡 Gemini AI Event Optimizer")
    st.markdown("Get AI-powered insights to enhance your event strategies.")

    selected_event_id_ai = st.selectbox(
        "Select Event for AI Analysis:",
        options=["Overall Event Strategy"] + [e["id"] + " - " + e["name"] for e in st.session_state.events if e["status"] != "Planning"],
        key="sel_event_ai"
    )

    if st.button("🤖 Analyze and Suggest Optimizations", key="gemini_event_opt_btn"):
        prompt_context = "Current Event Management System State:\n"
        prompt_context += f"- System Date: {st.session_state.system_time.strftime('%Y-%m-%d')}\n"
        
        event_to_analyze = None
        if selected_event_id_ai != "Overall Event Strategy":
            evt_id_only = selected_event_id_ai.split(" - ")[0]
            event_to_analyze = next((e for e in st.session_state.events if e["id"] == evt_id_only), None)

        if event_to_analyze:
            metrics = st.session_state.event_performance_metrics.get(event_to_analyze["id"],{})
            prompt_context += f"\nAnalyzing Specific Event: {event_to_analyze['name']} (ID: {event_to_analyze['id']})\n"
            prompt_context += f"  Type: {event_to_analyze['type']}, Status: {event_to_analyze['status']}\n"
            prompt_context += f"  Event Date: {event_to_analyze['event_date'].strftime('%Y-%m-%d')}\n"
            prompt_context += f"  Goals: {', '.join(event_to_analyze['goals'])}\n"
            prompt_context += f"  Target Audience: {event_to_analyze['target_audience_description']}\n"
            prompt_context += f"  Content Highlights: {event_to_analyze['content_highlights']}\n"
            prompt_context += f"  Performance Metrics:\n"
            prompt_context += f"    Registrations: {metrics.get('registrations',0)}\n"
            prompt_context += f"    Attendance: {metrics.get('attendance',0)} (Show-up: {(metrics.get('attendance',0)/(metrics.get('registrations',1) or 1)*100):.1f}%)\n"
            prompt_context += f"    Leads Generated: {metrics.get('leads_generated',0)}\n"
            prompt_context += f"    Cost Incurred: ${metrics.get('cost_incurred',0):.0f}\n"
            prompt_context += f"    Est. Revenue Impact: ${metrics.get('estimated_revenue_impact',0):.0f}\n"
        else: # Overall Strategy
            prompt_context += f"Analyzing Overall Event Strategy. Total Events: {len(st.session_state.events)}\n"
            # Could summarize top 2-3 events here if many.

        prompt_context += "\nRecent System Activity (last 5 logs):\n"
        for log_item in st.session_state.system_log_event[-5:]:
            prompt_context += f"- {log_item}\n"
        
        final_gemini_prompt = (
            f"{prompt_context}\n\n"
            "Based on this information, provide:\n"
            "1. A brief summary of the current event situation/performance.\n"
            "2. If a specific event is analyzed: Three concrete, actionable recommendations to improve its success (registrations, attendance, ROI). Be specific (e.g., 'For the 'Tech Summit', consider a targeted email to 'AI' interest segment with a special offer for early sign-ups.').\n"
            "3. If 'Overall Event Strategy' is selected: Three strategic recommendations for the entire event program (e.g., 'Explore co-marketing webinars with partners in the 'Finance' sector to expand reach', or 'Standardize post-event lead nurturing sequence.').\n"
            "4. Suggest one creative idea for promotional content (e.g., a catchy subject line, a social media post angle, or a unique selling proposition) for the selected event or for a typical upcoming event if overall strategy.\n"
            "5. Identify one potential risk or missed opportunity and how to address it."
        )
        
        with st.spinner("Gemini is brainstorming event strategies..."):
            st.session_state.ai_event_recommendations = get_gemini_event_response(final_gemini_prompt, temperature=0.8, max_tokens=800)
        
        if st.session_state.ai_event_recommendations:
            st.success("AI Recommendations Generated!")
        else:
            st.error("Failed to get recommendations from Gemini.")

    if st.session_state.ai_event_recommendations:
        st.markdown("---")
        st.markdown(st.session_state.ai_event_recommendations)


# Auto-run simulation loop
if st.session_state.simulation_running_event:
    if st.session_state.events and st.session_state.audience_pool:
        simulate_event_campaign_step()
        time.sleep(1.0 / simulation_speed_event)
        st.rerun()
    else:
        log_event_activity("Paused: Ensure events and audience pool exist.")
        st.session_state.simulation_running_event = False
        st.rerun()
