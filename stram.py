import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import folium
from streamlit_folium import folium_static
import googlemaps
from geopy.distance import geodesic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_mistralai import ChatMistralAI
import httpx

# Configuration
CSV_PATH = r"Data\pdf_folder\geocoded_data.csv"

load_dotenv(dotenv_path="./credentials.env")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")



# Custom CSS for enhanced styling
def apply_custom_styling():
    st.markdown("""
    <style>
    .title { 
        font-size: 2.5rem; 
        color: #2c3e50; 
        text-align: center; 
        margin-bottom: 20px; 
        font-weight: bold; 
    }
    .carrier-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    .carrier-card:hover {
        transform: scale(1.03);
    }
    </style>
    """, unsafe_allow_html=True)

def load_geocoded_data():
    """Load geocoded carrier data from CSV."""
    try:
        return pd.read_csv(CSV_PATH, encoding='ISO-8859-1')
    except Exception as e:
        st.error(f"Error loading carrier data: {e}")
        return pd.DataFrame()

def init_mistral_chat_model():
    """Initialize Mistral chat model with error handling."""
    try:
        return ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model="mistral-large-latest",
            temperature=0.7,
            http_client=httpx.Client(timeout=30.0)
        )
    except Exception as e:
        st.error(f"Failed to initialize Mistral API: {str(e)}")
        return None

def create_chat_prompt():
    """Create chat prompt template for logistics recommendations."""
    return ChatPromptTemplate.from_messages([
        ("system", """
        You are a Logistics Recommendation Expert specializing in location-based carrier matching. 
        Given the chat history and latest question, create a comprehensive carrier recommendation.

        Recommendation Guidelines:
        1. Analyze the logistics requirements carefully
        2. Suggest top 3 most suitable carriers
        3. For each carrier, provide:
           - Company name
           - Recommended service details
           - Potential suitability for the specific need
        4. Explain carrier matching logic
        5. Provide insights into location and service capabilities

        If details are insufficient, ask clarifying questions about:
        - Type of goods
        - Shipment weight/volume
        - Special handling requirements
        - Specific transportation needs
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

def get_closest_carriers(user_location, geocoded_data, n=3):
    """Find closest carriers based on user location."""
    try:
        user_coords = (user_location['lat'], user_location['long'])
        
        def calculate_distance(row):
            try:
                return geodesic(user_coords, (row['lat'], row['long'])).kilometers
            except Exception:
                return float('inf')
        
        geocoded_data['distance'] = geocoded_data.apply(calculate_distance, axis=1)
        closest_carriers = geocoded_data.nsmallest(n, 'distance')
        
        return closest_carriers[['carrier_name', 'state', 'address', 'lat', 'long', 'distance']]
    except Exception as e:
        st.error(f"Error finding closest carriers: {e}")
        return pd.DataFrame()

def generate_interactive_map(carriers_data, user_location):
    """Generate interactive map with carrier locations."""
    try:
        center_lat = user_location['lat']
        center_lng = user_location['long']
        
        m = folium.Map(location=[center_lat, center_lng], zoom_start=10)
        
        # User location marker
        folium.Marker(
            [user_location['lat'], user_location['long']],
            popup="Your Location",
            icon=folium.Icon(color="red", icon="home"),
            tooltip="Your Location"
        ).add_to(m)
        
        # Carrier markers
        for _, row in carriers_data.iterrows():
            popup_text = f"""
                <b>{row['carrier_name']}</b><br>
                State: {row['state']}<br>
                Address: {row['address']}<br>
                Distance: {row['distance']:.2f} km
            """
            folium.Marker(
                [row['lat'], row['long']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color="blue", icon="truck"),
                tooltip=row['carrier_name']
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Map generation error: {e}")
        return None

def main():
    # Page configuration
    st.set_page_config(page_title="LogisticsLink AI", layout="wide")
    apply_custom_styling()

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", 
        ["Carrier Finder", "Get Instant Logistics Support", "Company Registration"]
    )

    # Load carrier data and initialize Mistral model
    geocoded_data = load_geocoded_data()
    chat_model = init_mistral_chat_model()

    if page == "Carrier Finder":
        st.markdown('<div class="title">🚚 Find Nearest Logistics Carriers</div>', unsafe_allow_html=True)
        
        # Initialize Google Maps client
        gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

        # Address input
        col1, col2 = st.columns([3,1])
        with col1:
            user_address = st.text_input("Enter your full address:", 
                                         placeholder="123 Main St, City, State, ZIP")
        with col2:
            st.write("") 
            find_carriers = st.button("🔍 Find Carriers")
        
        if find_carriers and user_address:
            with st.spinner('Finding nearest carriers...'):
                try:
                    # Geocode user address
                    geocode_result = gmaps_client.geocode(user_address)
                    if not geocode_result:
                        st.error("Could not geocode the provided address.")
                        return

                    user_location = {
                        'lat': geocode_result[0]['geometry']['location']['lat'],
                        'long': geocode_result[0]['geometry']['location']['lng']
                    }

                    # Find closest carriers
                    closest_carriers = get_closest_carriers(user_location, geocoded_data)
                    
                    if not closest_carriers.empty:
                        st.subheader("Closest Carriers")
                        
                        # Display carriers in styled cards
                        for _, carrier in closest_carriers.iterrows():
                            st.markdown(f"""
                            <div class="carrier-card">
                                <h3>{carrier['carrier_name']}</h3>
                                <p>
                                <strong>Location Details:</strong><br>
                                • State: {carrier['state']}<br>
                                • Address: {carrier['address']}<br>
                                • Distance: {carrier['distance']:.2f} km
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Generate interactive map
                        m = generate_interactive_map(closest_carriers, user_location)
                        if m:
                            folium_static(m)
                    else:
                        st.warning("No carriers found near your location.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif page == "Get Instant Logistics Support":
        st.title("🤖 Get Instant Logistics Support")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Welcome message
        if len(st.session_state.messages) == 0:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "👋 Hello! I'm your Logistics AI Assistant. Describe your shipping requirements, and I'll help you find the best carrier solution."
            })
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Describe your logistics needs..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Analyzing your logistics requirements..."):
                try:
                    # Prepare chat prompt
                    chat_prompt = create_chat_prompt()
                    
                    # Generate response using Mistral model
                    response = chat_model.invoke(
                        chat_prompt.format_messages(
                            input=prompt,
                            chat_history=st.session_state.messages
                        )
                    )
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.content
                    })
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(response.content)
                
                except Exception as e:
                    st.error(f"Error generating recommendation: {str(e)}")

    elif page == "Company Registration":
        st.title("🏢 Register Your Logistics Company")
        with st.form("company_registration"):
            company_name = st.text_input("Company Name")
            company_type = st.selectbox("Company Type", 
                ["Trucking", "Warehousing", "Last Mile Delivery", "Full Service"]
            )
            areas_served = st.multiselect("Areas Served", 
                ["Lagos", "Abuja", "Port Harcourt", "Kano", "Ibadan"]
            )
            vehicle_types = st.multiselect("Vehicle Types", 
                ["Small Van", "Medium Truck", "Large Truck", "Refrigerated Truck"]
            )
            contact_email = st.text_input("Contact Email")
            phone_number = st.text_input("Phone Number")
            company_description = st.text_area("Company Description")
            submitted = st.form_submit_button("Register Company")
            
            if submitted:
                st.success("Company registered successfully! Our team will review your application.")

if __name__ == "__main__":
    main()