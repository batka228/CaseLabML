from django.shortcuts import render
from .forms import ReviewForm
from .utils import predict_sentiment

def analyze_review(request):
    result = None

    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['review_text']
            rating = predict_sentiment(review_text)

            # Определение позитивности или негативности
            if rating >= 7:
                sentiment = "Положительный"
            elif rating <= 4:
                sentiment = "Отрицательный"
            else:
                sentiment = "Нейтральный"

            result = {
                'sentiment': sentiment,
                'rating': round(rating, 2)
            }
    else:
        form = ReviewForm()

    return render(request, 'sentiment/analyze.html', {'form': form, 'result': result})
