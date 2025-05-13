import pandas as pd
import numpy as np
from collections import Counter
import emoji
import re
from wordcloud import WordCloud

# URL extraction function without external dependencies
def extract_urls(text):
    """Extract URLs from text using regex"""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    return re.findall(url_pattern, text)

def fetch_stats(selected_user, df):
    """
    Extract basic statistics from the chat data
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        tuple: (number of messages, number of words, number of media messages, number of links)
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Total messages
    num_messages = df.shape[0]
    
    # Total words
    words = []
    for message in df['message']:
        if isinstance(message, str):  # Check if message is a string
            words.extend(message.split())
    
    # Media messages (check for different formats)
    media_patterns = ['<Media omitted>', '<Media omitted>\n']
    num_media_messages = df[df['message'].isin(media_patterns)].shape[0]
    
    # Links shared
    links = []
    for message in df['message']:
        if isinstance(message, str):  # Check if message is a string
            links.extend(extract_urls(message))
    
    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    """
    Find the most active users in the chat
    
    Args:
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        tuple: (Series with user message counts, DataFrame with percentage stats)
    """
    if df.empty or 'user' not in df.columns:
        return pd.Series(), pd.DataFrame(columns=['User', 'Percent'])
    
    # Drop group notifications
    chat_df = df[df['user'] != 'group_notification']
    
    if chat_df.empty:
        return pd.Series(), pd.DataFrame(columns=['User', 'Percent'])
    
    # Count messages by user
    x = chat_df['user'].value_counts().head(10)
    
    # Calculate percentages
    df_percent = round((chat_df['user'].value_counts() / chat_df.shape[0]) * 100, 2).reset_index()
    df_percent.columns = ['User', 'Percent']
    
    return x, df_percent

def create_wordcloud(selected_user, df):
    """
    Generate a wordcloud from chat messages
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        WordCloud: Generated wordcloud object
    """
    if df.empty or 'user' not in df.columns or 'message' not in df.columns:
        # Return an empty WordCloud if data is invalid
        return WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(" ")
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Drop group notifications
    df = df[df['user'] != 'group_notification']
    
    # Remove media messages (check both formats)
    media_patterns = ['<Media omitted>', '<Media omitted>\n']
    df = df[~df['message'].isin(media_patterns)]
    
    if df.empty:
        return WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(" ")
    
    # Combine all messages, handling non-string values
    text = " ".join([msg for msg in df['message'] if isinstance(msg, str)])
    
    # Generate wordcloud
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(text)
    
    return df_wc

def monthly_timeline(selected_user, df):
    """
    Generate monthly timeline of messages
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        pd.DataFrame: Timeline with message counts by month and year
    """
    if df.empty or not all(col in df.columns for col in ['user', 'year', 'month', 'message']):
        return pd.DataFrame(columns=['year', 'month', 'message', 'time'])
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Group by month and year
    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()
    
    # Create a unified time column for plotting
    timeline['time'] = timeline['month'] + ' ' + timeline['year'].astype(str)
    
    return timeline

def daily_timeline(selected_user, df):
    """
    Generate daily timeline of messages
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        pd.DataFrame: Timeline with message counts by date
    """
    if df.empty or not all(col in df.columns for col in ['user', 'only_date', 'message']):
        return pd.DataFrame(columns=['only_date', 'message'])
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Group by date
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    
    return daily_timeline

def week_activity_map(selected_user, df):
    """
    Count messages by day of week
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        pd.Series: Message counts by day of week
    """
    if df.empty or 'day_name' not in df.columns:
        return pd.Series()
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Sort by weekday (Monday to Sunday)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_name'].value_counts().reindex(weekday_order, fill_value=0)
    
    return day_counts

def month_activity_map(selected_user, df):
    """
    Count messages by month
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        pd.Series: Message counts by month
    """
    if df.empty or 'month' not in df.columns:
        return pd.Series()
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Sort by month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month'].value_counts().reindex(month_order, fill_value=0)
    
    return month_counts

def activity_heatmap(selected_user, df):
    """
    Create a heatmap of activity by day of week and hour period
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        pd.DataFrame: Pivot table with counts by day and hour period
    """
    if df.empty or not all(col in df.columns for col in ['user', 'day_name', 'period', 'message']):
        # Return empty DataFrame with proper structure for heatmap
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return pd.DataFrame(index=weekday_order)
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Weekday ordering for better visualization
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot_table = df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)
    
    # Reindex with proper day order
    pivot_table = pivot_table.reindex(weekday_order)
    
    return pivot_table

def most_common_words(selected_user, df):
    """
    Find the most commonly used words in the chat
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with word frequencies
    """
    if df.empty or 'message' not in df.columns:
        return pd.DataFrame(columns=[0, 1])
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Filter out media messages and group notifications
    df = df[df['user'] != 'group_notification']
    media_patterns = ['<Media omitted>', '<Media omitted>\n']
    df = df[~df['message'].isin(media_patterns)]
    
    # Define stopwords to filter out common words
    stopwords = set([
        'the', 'to', 'and', 'is', 'in', 'it', 'of', 'for', 'on', 'that', 'this', 
        'was', 'with', 'as', 'at', 'by', 'an', 'be', 'are', 'or', 'from', 'had',
        'have', 'has', 'a', 'i', 'you', 'me', 'we', 'my', 'your', 'our', 'he', 'she',
        'him', 'her', 'they', 'their', 'am', 'if', 'but', 'so', 'not', 'what', 'when',
        'where', 'who', 'how', 'why', 'which', 'ok', 'okay', 'yes', 'no', 'can', 'will',
        'just', 'do', 'did', 'done', 'going', 'go', 'went', 'gone', 'get', 'got', 'getting'
    ])
    
    words = []
    for message in df['message']:
        if isinstance(message, str):
            # Clean the message and split into words
            clean_message = re.sub(r'[^\w\s]', ' ', message.lower())
            message_words = [word for word in clean_message.split() if word.isalpha() and word not in stopwords]
            words.extend(message_words)
    
    # Get most common words
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    
    if most_common_df.empty:
        return pd.DataFrame(columns=[0, 1])
    
    return most_common_df

def emoji_helper(selected_user, df):
    """
    Extract emoji usage statistics
    
    Args:
        selected_user (str): User to filter by, or 'Overall' for all users
        df (pd.DataFrame): Preprocessed chat DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with emoji frequencies
    """
    if df.empty or 'message' not in df.columns:
        return pd.DataFrame(columns=[0, 1])
    
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    emojis = []
    for message in df['message']:
        if isinstance(message, str):
            # Extract emojis from message
            emoji_list = [c for c in message if c in emoji.EMOJI_DATA]
            emojis.extend(emoji_list)
    
    # Get emoji counts
    emoji_counter = Counter(emojis)
    
    # Create DataFrame with emoji counts
    emoji_df = pd.DataFrame(emoji_counter.most_common(10))
    
    if emoji_df.empty:
        return pd.DataFrame(columns=[0, 1])
    
    return emoji_df