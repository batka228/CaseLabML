import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import DistilBertTokenizer

class SentimentRegressor(nn.Module):
    def __init__(self, pretrained_model, dropout=0.2):
        super(SentimentRegressor, self).__init__()
        self.bert = pretrained_model
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        x = self.fc1(pooled_output)
        x = self.relu(x)

        x = self.fc2(x)

        return x
    



pretrained_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = SentimentRegressor(pretrained_model=pretrained_model)

model.load_state_dict(torch.load("~/model/sentiment_regressor.pth", map_location=torch.device('cpu')))



model_path = "~/model"


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