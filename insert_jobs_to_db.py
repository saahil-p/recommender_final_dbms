import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

def upload_courses_to_mysql():
    # Read the CSV file
    df = pd.read_csv('job_data_filtered.csv')
    
    # Add stemmed descriptions
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import nltk
    
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Define stemming function
    def stem_text(text):
        if isinstance(text, str):
            words = nltk.word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            return ' '.join(stemmer.stem(word) for word in filtered_words)
        return ""
    
    # Add stemmed descriptions
    df['job_description_stemmed'] = df['job_description'].apply(stem_text)
    
    # Create SQLAlchemy engine
    engine = create_engine('mysql+mysqlconnector://root:Saahil123@localhost:3306/Project_final')
    
    try:
        # Upload dataframe to MySQL
        df.to_sql('jobs_final', 
                 engine, 
                 if_exists='replace',  # Replace if table exists
                 index=True,  # Use index as primary key
                 index_label='id',  # Name the index column 'id'
                 chunksize=1000)  # Upload in chunks to handle large datasets
        
        print("Successfully uploaded courses to MySQL database!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        engine.dispose()

if __name__ == "__main__":
    upload_courses_to_mysql()