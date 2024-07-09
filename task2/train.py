{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c6d7613a-7ece-40a1-ba20-5239d5d0be26",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validated RMSE scores: [0.00383989 0.00378737 0.0037784  0.00377849 0.00379543]\n",
      "Mean RMSE: 0.003795917255571311\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.linear_model import Ridge, Lasso\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import joblib\n",
    "import argparse\n",
    "\n",
    "file_path = r'C:\\Users\\Max Nazarenko\\Desktop\\New folder\\train.csv'\n",
    "df = pd.read_csv(file_path)\n",
    "df1 = df.drop(columns=['8'])\n",
    "    \n",
    "def train_model(model_type,df):\n",
    "    X_train = df.drop('target', axis=1)\n",
    "    y_train = df['target']\n",
    "\n",
    "    if model_type == 'random_forest':\n",
    "        model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "    elif model_type == 'ridge':\n",
    "        model = Ridge(alpha=alpha, random_state=42)\n",
    "    elif model_type == 'lasso':\n",
    "        model = Lasso(alpha=alpha, random_state=42)\n",
    "    else:\n",
    "        raise ValueError(\"Invalid model type. Choose from 'random_forest', 'ridge', or 'lasso'.\")\n",
    "\n",
    "    # Cross-validation and model evaluation\n",
    "    scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)\n",
    "    rmse_scores = -scores\n",
    "    print(f\"Cross-validated RMSE scores: {rmse_scores}\")\n",
    "    print(f\"Mean RMSE: {rmse_scores.mean()}\")\n",
    "    model.fit(X_train, y_train)\n",
    "    return model\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    model_type = 'random_forest'\n",
    "    alpha = 1.0\n",
    "    \n",
    "    trained_model = train_model(model_type, df1)\n",
    "    \n",
    "    # Сохранение модели\n",
    "    joblib.dump(trained_model, 'models/model.joblib')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5921fe2b-291d-49ed-8d72-7257496864c2",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
