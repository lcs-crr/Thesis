"""
Lucas Correia
LIACS | Leiden University
Einsteinweg 55 | 2333 CC Leiden | The Netherlands
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()

# create dataset with 5 predictor variables
X, y = datasets.make_classification(
    n_samples=1000, n_features=4, n_informative=3, n_redundant=1, random_state=0
)

# split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# fit logistic regression model to dataset
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# use logistic regression model to make predictions
y_score = classifier.predict_proba(X_test)[:, 1]

# calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y_test, y_score)

# Create Precision-Recall Curve using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=recall, y=precision, line=dict(color="black")))

fig.update_xaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    title_text="Precision [-]",
    range=[-0.01, 1.01],
)
fig.update_yaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    title_text="Recall [-]",
    range=[-0.01, 1.01],
)

fig.update_layout(
    showlegend=False,
    plot_bgcolor="white",
    width=600,
    height=600,
    font_color="black",
)

fig.write_image(os.path.join(os.environ["asset_path"], "precision_recall_curve.svg"))

fig.show()
