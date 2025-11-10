import streamlit as st
import requests
import json

st.title("Lore ML Recommendation Engine")

st.write("Enter the conversation JSON below to get recommendations.")

# Text area for JSON input
json_input = st.text_area("Conversation JSON", height=300)

if st.button("Get Recommendations"):
    if json_input:
        try:
            # Parse the JSON input
            conversation_data = json.loads(json_input)
            
            # Send the request to the API
            response = requests.post("http://api:8000/v1/recommend", json=conversation_data)
            
            if response.status_code == 200:
                # Display the recommendations
                st.success("Successfully received recommendations!")
                
                recommendations = response.json()
                
                st.subheader("Recommendations")
                for rec in recommendations.get("recommendations", []):
                    st.write(f"**Post ID:** {rec['post_id']}")
                    st.write(f"**Score:** {rec['score']}")
                    st.write(f"**Text:** {rec['text']}")
                    st.markdown("---")
                
                st.subheader("Debug Information")
                st.json(recommendations.get("debug", {}))
                
            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.write(response.text)
                
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your input.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API: {e}")
    else:
        st.warning("Please enter some JSON data.")
