# Credit Scoring Model for Bati Bank

## Project Overview

This project develops a credit scoring model for Bati Bank's buy-now-pay-later service in partnership with an eCommerce platform. The model uses customer transaction data to predict credit risk, assign credit scores, and recommend optimal loan amounts and durations.

## Credit Scoring Business Understanding
### Influence of Basel II Accord on Model Interpretability

The Basel II Capital Accord emphasizes robust risk measurement and management, requiring financial institutions to maintain transparent and auditable credit risk models. This influences our need for an interpretable and well-documented model to ensure compliance with regulatory standards. An interpretable model, such as Logistic Regression with Weight of Evidence (WoE), allows regulators to understand how features contribute to risk predictions, ensuring fairness and accountability. Comprehensive documentation of data processing, feature engineering, and model decisions is critical to demonstrate adherence to Basel II's risk assessment guidelines and to facilitate audits.

### Necessity and Risks of Proxy Variables

Since the dataset lacks a direct "default" label, we create a proxy variable (is_high_risk) based on customer behavior, specifically Recency, Frequency, and Monetary (RFM) metrics. This proxy is necessary to categorize customers as high or low risk, enabling model training. However, using a proxy introduces business risks, such as misclassification if the RFM-based clusters do not accurately reflect default likelihood. For example, a customer with low transaction frequency might be labeled high-risk but could still repay loans reliably, leading to lost business opportunities or unfair rejections. Regular validation against real-world outcomes and iterative refinement of the proxy definition are essential to mitigate these risks.

### Trade-offs Between Simple and Complex Models

In a regulated financial context, choosing between simple, interpretable models (e.g., Logistic Regression with WoE) and complex, high-performance models (e.g., Gradient Boosting) involves key trade-offs. Simple models are easier to interpret, explain to regulators, and align with Basel II's transparency requirements, but they may sacrifice predictive accuracy, potentially leading to suboptimal risk assessments. Complex models like Gradient Boosting can capture non-linear patterns and improve prediction accuracy, enhancing loan approval decisions. However, their "black-box" nature makes regulatory justification challenging, and they require more computational resources. We prioritize interpretability for regulatory compliance but explore complex models to benchmark performance, selecting the best model based on a balance of accuracy, interpretability, and regulatory requirements.
