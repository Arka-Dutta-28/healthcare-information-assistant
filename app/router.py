from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter

encoder = HuggingFaceEncoder(
    name="sentence-transformers/all-mpnet-base-v2"
)

faq_tips = Route(
    name='faq-tips',
    utterances=[
        "What are the common signs of a cold?",
        "How much physical activity is recommended per week?",
        "What foods are best for boosting the immune system?",
        "How many hours of sleep do adults typically need?",
        "Why is staying hydrated important for overall health?",
        "Cold symptoms",
        "Signs I have a cold",
        "Flu symptoms",
        "How do I know if I have a cold?",
        "What are the symptoms of flu?",
        "How often should I exercise?",
        "Recommended physical activity per week",
        "Weekly exercise guidelines",
        "How to improve sleep quality?",
        "Diet tips for a healthy immune system",
        "Healthy lifestyle tips",
        "How to avoid disease?",
        "Disease prevention tips",
    ]
)

appointment = Route(
    name='appointment',
    utterances=[
        "What are the visiting hours for the cardiology department?",
        "How can I schedule a dental check-up at City Health Clinic?",
        "What is the emergency contact number for the downtown hospital?",
        "Is online appointment booking available for the pediatric unit?",
        "Where can I find the phone number for the hospital pharmacy?",
        "Visiting hours at the medical center",
        "When can I visit cardiology?",
        "Opening times for hospital departments",
        "Hospital visiting schedule"
    ]
)

# Create the index with routes passed during initialization
routes=[faq_tips, appointment]

# Initialize the router with the index
router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

if __name__ == "__main__":
    print(router("what is the visiting time in medical centre?").name)
    print(router("Symptoms of bronchitis").name)
    print(router("How often should I exercise?").name)
    # print(router.routes)