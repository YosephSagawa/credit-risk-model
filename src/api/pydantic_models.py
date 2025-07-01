from pydantic import BaseModel
from datetime import datetime

class CustomerData(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: int
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: int
    TransactionStartTime: datetime
    PricingStrategy: int
    FraudResult: int

class PredictionResponse(BaseModel):
    probability: float