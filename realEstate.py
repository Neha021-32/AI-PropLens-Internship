from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import pandas as pd

# Sample dataset for real estate projects in Gurgaon
projects_data = [
    {
        'name': 'DLF Phase 5',
        'lat': 28.4595,
        'lng': 77.0266,
        'rating': 4.5,
        'reviews': [
            {'text': 'Great amenities and location.'},
            {'text': 'Spacious apartments and good security.'}
        ],
        'address': 'DLF Phase 5, Gurgaon',
        'amenities': ['Swimming Pool', 'Gym', 'Park']
    },
    {
        'name': 'Sohna Road Residency',
        'lat': 28.4675,
        'lng': 77.0297,
        'rating': 4.0,
        'reviews': [
            {'text': 'Well connected to main roads.'},
            {'text': 'Good maintenance and facilities.'}
        ],
        'address': 'Sohna Road, Gurgaon',
        'amenities': ['Clubhouse', 'Parking', 'Play Area']
    },
    {
        'name': 'Golf Course Extension Apartments',
        'lat': 28.4596,
        'lng': 77.0726,
        'rating': 3.8,
        'reviews': [
            {'text': 'Affordable housing with decent amenities.'},
            {'text': 'Needs better maintenance.'}
        ],
        'address': 'Golf Course Extension, Gurgaon',
        'amenities': ['Jogging Track', 'Security', 'Power Backup']
    }
]

def extract_amenities(reviews):
    review_texts = [review['text'] for review in reviews]
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(review_texts)
    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(dtm)
    
    topic_words = vectorizer.get_feature_names_out()
    topic_components = lda.components_[0]
    top_words = topic_words[topic_components.argsort()[-10:]]
    
    return top_words

def analyze_sentiment(reviews):
    sentiments = [TextBlob(review['text']).sentiment.polarity for review in reviews]
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return average_sentiment

def compare_projects(target_project, projects):
    comparisons = []
    
    for project in projects:
        amenities = extract_amenities(project['reviews'])
        sentiment = analyze_sentiment(project['reviews'])
        
        comparisons.append({
            'name': project['name'],
            'address': project['address'],
            'rating': project['rating'],
            'amenities': amenities,
            'sentiment': sentiment
        })
    
    return comparisons

# Example usage
target_project = {
    'name': 'Central Park',
    'lat': 28.4594,
    'lng': 77.0267,
    'address': 'Sector 48, Gurgaon'
}

project_comparisons = compare_projects(target_project, projects_data)

# Create a DataFrame for tabular output
df = pd.DataFrame(project_comparisons)

# Display the DataFrame
print(df)

# Optional: if you want to save the output to a file, e.g., 'project_comparisons.csv'
df.to_csv('project_comparisons.csv',index=False)