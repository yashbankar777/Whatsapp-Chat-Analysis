import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide"
)

# Main title
st.title("WhatsApp Chat Analyzer")

# Upload file
uploaded_file = st.sidebar.file_uploader("Choose a WhatsApp chat export file", type=['txt'])

if uploaded_file is not None:
    # Read file as string
    bytes_data = uploaded_file.getvalue()
    try:
        # Convert bytes to string
        data = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            # Try with different encoding if utf-8 fails
            data = bytes_data.decode("latin-1")
        except Exception as e:
            st.error(f"Error decoding the file: {e}")
            data = None

    if data:
        # Preprocess the data
        df = preprocessor.preprocess(data)
        
        if df.empty:
            st.error("Could not process the chat data. Please make sure you've uploaded a valid WhatsApp chat export file.")
        else:
            # Display dataframe sample
            with st.expander("Show raw data sample"):
                st.dataframe(df.head())
            
            # Get unique users
            user_list = df['user'].unique().tolist()
            
            # Remove group_notification from the list if present
            if 'group_notification' in user_list:
                user_list.remove('group_notification')
            
            # Sort the list alphabetically
            user_list.sort()
            
            # Add Overall option at the beginning
            user_list.insert(0, "Overall")
            
            # User selector
            selected_user = st.sidebar.selectbox("Select User", user_list)
            
            # Show analysis button
            if st.sidebar.button("Show Analysis"):
                
                # Stats
                st.header("Chat Statistics")
                num_messages, num_words, num_media, num_links = helper.fetch_stats(selected_user, df)
                
                # Create four columns for stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Messages", num_messages)
                
                with col2:
                    st.metric("Total Words", num_words)
                    
                with col3:
                    st.metric("Media Shared", num_media)
                    
                with col4:
                    st.metric("Links Shared", num_links)
                
                # Monthly timeline
                st.header("Monthly Activity")
                timeline = helper.monthly_timeline(selected_user, df)
                if not timeline.empty:
                    fig, ax = plt.subplots()
                    ax.plot(timeline['time'], timeline['message'], color='green', marker='o')
                    plt.xticks(rotation='vertical')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No monthly activity data available.")
                
                # Daily timeline
                st.header("Daily Activity")
                daily_timeline = helper.daily_timeline(selected_user, df)
                if not daily_timeline.empty:
                    fig, ax = plt.subplots()
                    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black', marker='o')
                    plt.xticks(rotation='vertical')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No daily activity data available.")
                
                # Activity map
                st.header("Activity Patterns")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Weekly Activity")
                    week_activity = helper.week_activity_map(selected_user, df)
                    if not week_activity.empty:
                        fig, ax = plt.subplots()
                        ax.bar(week_activity.index, week_activity.values, color='purple')
                        plt.xticks(rotation='vertical')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("No weekly activity data available.")
                
                with col2:
                    st.subheader("Monthly Activity")
                    month_activity = helper.month_activity_map(selected_user, df)
                    if not month_activity.empty:
                        fig, ax = plt.subplots()
                        ax.bar(month_activity.index, month_activity.values, color='orange')
                        plt.xticks(rotation='vertical')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("No monthly activity data available.")
                
                # Activity heatmap
                st.header("Weekly Activity Heatmap")
                heatmap_data = helper.activity_heatmap(selected_user, df)
                if not heatmap_data.empty and not heatmap_data.columns.empty:
                    fig, ax = plt.subplots(figsize=(15, 6))
                    sns.heatmap(heatmap_data, cmap='Greens', linewidths=0.5, ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Not enough data to generate activity heatmap.")
                
                # User activity comparison - Only show for Overall analysis
                if selected_user == 'Overall':
                    st.header("Most Active Users")
                    user_counts, percent_df = helper.most_busy_users(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if not user_counts.empty:
                            fig, ax = plt.subplots()
                            ax.bar(user_counts.index, user_counts.values, color='red')
                            plt.xticks(rotation='vertical')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.info("No user activity data available.")
                    
                    with col2:
                        if not percent_df.empty:
                            st.dataframe(percent_df)
                        else:
                            st.info("No percentage data available.")
                
                # Word cloud
                st.header("Word Cloud")
                try:
                    df_wc = helper.create_wordcloud(selected_user, df)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    plt.axis('off')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating word cloud: {e}")
                
                # Most common words
                st.header("Most Common Words")
                most_common_df = helper.most_common_words(selected_user, df)
                if not most_common_df.empty and most_common_df.shape[1] >= 2:
                    # Create a readable dataframe with named columns
                    word_freq_df = pd.DataFrame({
                        'Word': most_common_df[0],
                        'Count': most_common_df[1]
                    })
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.barh(word_freq_df['Word'], word_freq_df['Count'], color='blue')
                    plt.xlabel('Frequency')
                    plt.ylabel('Words')
                    plt.xticks(rotation='vertical')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Also show as a table
                    with st.expander("Show data as table"):
                        st.dataframe(word_freq_df)
                else:
                    st.info("No word frequency data available.")
                
                # Emoji analysis
                st.header("Emoji Analysis")
                emoji_df = helper.emoji_helper(selected_user, df)
                if not emoji_df.empty and emoji_df.shape[1] >= 2:
                    # Create columns for emoji display and chart
                    col1, col2 = st.columns(2)
                    
                    # Create a readable dataframe with named columns
                    emoji_freq_df = pd.DataFrame({
                        'Emoji': emoji_df[0],
                        'Count': emoji_df[1]
                    })
                    
                    with col1:
                        st.dataframe(emoji_freq_df)
                    
                    with col2:
                        fig, ax = plt.subplots()
                        ax.pie(emoji_freq_df['Count'], labels=emoji_freq_df['Emoji'], autopct='%1.1f%%')
                        plt.tight_layout()
                        st.pyplot(fig)
                else:
                    st.info("No emoji data available.")
    
else:
    # Instructions when no file is uploaded
    st.info("""
    ### How to export your WhatsApp chat:
    1. Open WhatsApp on your phone
    2. Open the chat you want to analyze
    3. Tap the three dots (menu) in the top right
    4. Select 'More' > 'Export chat'
    5. Choose 'Without Media'
    6. Send the exported file to yourself
    7. Upload the file here
    """)
    
    # Display sample images or demo content
    st.write("### This tool will help you analyze:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("📊 Message patterns")
    with col2:
        st.write("🔤 Common words & phrases")
    with col3:
        st.write("😀 Emoji usage")