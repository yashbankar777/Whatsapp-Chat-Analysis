import re
import pandas as pd
import io

def preprocess(data):
    """
    Preprocess WhatsApp chat data into a structured pandas DataFrame.
    
    Args:
        data (str): Raw WhatsApp chat export text data
        
    Returns:
        pd.DataFrame: Structured DataFrame with message data and time-based features
    """
    if not isinstance(data, str) or not data.strip():
        print("Error: Input data is empty or not a string")
        return pd.DataFrame()
    
    # List of different WhatsApp date-time patterns to try
    patterns = [
        # Standard format: 25/04/23, 15:49 - Name: Message
        r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s[AP]M)?\s-\s',
        
        # Bracketed format: [04/25/23, 3:49:21 PM] Name: Message
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s[AP]M)?\]\s',
        
        # Date with dots: 25.04.2023, 15:49 - Name: Message
        r'(\d{1,2}\.\d{1,2}\.\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s[AP]M)?\s-\s',
        
        # Format with dashes: 2023-04-25, 15:49 - Name: Message
        r'(\d{4}-\d{1,2}-\d{1,2}),\s(\d{1,2}:\d{2}(?::\d{2})?)(?:\s[AP]M)?\s-\s'
    ]
    
    # Date formats to try for conversion
    date_formats = [
        '%d/%m/%Y, %H:%M', '%m/%d/%Y, %H:%M', 
        '%d/%m/%y, %H:%M', '%m/%d/%y, %H:%M',
        '%d/%m/%Y, %H:%M:%S', '%m/%d/%Y, %H:%M:%S',
        '%d/%m/%Y, %I:%M %p', '%m/%d/%Y, %I:%M %p',
        '%d.%m.%Y, %H:%M', '%d.%m.%y, %H:%M',
        '%Y-%m-%d, %H:%M', '%Y-%m-%d, %H:%M:%S'
    ]
    
    # Try each pattern
    for pattern_idx, pattern in enumerate(patterns):
        try:
            # Split the chat by the pattern
            splits = re.split(pattern, data)
            
            # Extract date and time matches
            date_time_matches = re.findall(pattern, data)
            
            if not date_time_matches or len(date_time_matches) == 0:
                continue
                
            # Build the messages and dates lists
            messages = []
            dates = []
            
            # We should have triplets in splits after pattern split: [before_match, date, time, message, ...]
            # First element is often empty/irrelevant text
            if len(splits) > 1:
                for i in range(0, len(splits) - 2, 3):
                    if i + 2 < len(splits):
                        date_part = splits[i + 1]
                        time_part = splits[i + 2]
                        message_part = splits[i + 3]
                        
                        dates.append(f"{date_part}, {time_part}")
                        messages.append(message_part)
            
            if len(dates) == 0 or len(messages) == 0:
                continue
                
            if len(dates) != len(messages):
                # Ensure lists have the same length
                min_len = min(len(dates), len(messages))
                dates = dates[:min_len]
                messages = messages[:min_len]
            
            # Create DataFrame
            df = pd.DataFrame({'message_date': dates, 'user_message': messages})
            
            # Try to convert dates with different formats
            datetime_conversion_success = False
            for fmt in date_formats:
                try:
                    df['date'] = pd.to_datetime(df['message_date'], format=fmt)
                    datetime_conversion_success = True
                    break
                except ValueError:
                    continue
            
            if not datetime_conversion_success:
                continue
            
            # Extract users and messages
            df['user'] = df['user_message'].str.extract(r'^(.*?):\s', expand=False)
            df['message'] = df['user_message'].str.replace(r'^.*?:\s', '', regex=True)
            
            # Handle group notifications and messages without a username
            mask = df['user'].isna()
            df.loc[mask, 'user'] = 'group_notification'
            df.loc[mask, 'message'] = df.loc[mask, 'user_message']
            
            # Clean up the DataFrame
            df = df.drop(columns=['message_date', 'user_message'])
            
            # Extract time-based features
            df['only_date'] = df['date'].dt.date
            df['year'] = df['date'].dt.year
            df['month_num'] = df['date'].dt.month
            df['month'] = df['date'].dt.month_name()
            df['day'] = df['date'].dt.day
            df['day_name'] = df['date'].dt.day_name()
            df['hour'] = df['date'].dt.hour
            df['minute'] = df['date'].dt.minute
            
            # Create hourly period labels
            df['period'] = df['hour'].apply(lambda h: f"{h:02d}-{(h+1)%24:02d}")
            
            # Check if we have a valid dataframe with required columns
            if len(df) > 0 and all(col in df.columns for col in ['date', 'user', 'message']):
                return df
            
        except Exception:
            continue
    
    # Alternative approach for different file formats
    try:
        # Try common CSV formats
        for delimiter in [',', '\t', ';']:
            try:
                df = pd.read_csv(io.StringIO(data), delimiter=delimiter)
                
                # Identify potential date, user and message columns
                date_col = None
                user_col = None
                msg_col = None
                
                for col in df.columns:
                    if col == df.columns[0] and df[col].dtype == 'object':
                        # First column is often the date in WhatsApp exports
                        date_col = col
                    elif col == df.columns[1] and df[col].dtype == 'object':
                        # Second column is often the user name
                        user_col = col
                    elif col == df.columns[2] and df[col].dtype == 'object':
                        # Third column is often the message content
                        msg_col = col
                
                if date_col and user_col and msg_col:
                    # Rename columns to standard format
                    df = df.rename(columns={date_col: 'date', user_col: 'user', msg_col: 'message'})
                    
                    # Only keep necessary columns
                    df = df[['date', 'user', 'message']]
                    
                    # Try to convert date column
                    for fmt in date_formats:
                        try:
                            df['date'] = pd.to_datetime(df['date'], format=fmt)
                            break
                        except:
                            continue
                    
                    # Extract time-based features
                    df['only_date'] = df['date'].dt.date
                    df['year'] = df['date'].dt.year
                    df['month_num'] = df['date'].dt.month
                    df['month'] = df['date'].dt.month_name()
                    df['day'] = df['date'].dt.day
                    df['day_name'] = df['date'].dt.day_name()
                    df['hour'] = df['date'].dt.hour
                    df['minute'] = df['date'].dt.minute
                    
                    # Create hourly period labels
                    df['period'] = df['hour'].apply(lambda h: f"{h:02d}-{(h+1)%24:02d}")
                    
                    if len(df) > 0:
                        return df
            except:
                continue
    except:
        pass
    
    return pd.DataFrame()  # Return empty DataFrame if all methods fail