import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import DistilBertTokenizer

class SentimentRegressor(nn.Module):
    def __init__(self, pretrained_model, dropout=0.3):
        super(SentimentRegressor, self).__init__()
        self.bert = pretrained_model
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output)
    



pretrained_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = SentimentRegressor(pretrained_model=pretrained_model)

model.load_state_dict(torch.load("/home/oleg/model/sentiment_regressor.pth", map_location=torch.device('cpu')))



model_path = "/home/oleg/model"


tokenizer = DistilBertTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


model.eval()
def predict_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        predicted_rating = outputs.item()

    return predicted_rating