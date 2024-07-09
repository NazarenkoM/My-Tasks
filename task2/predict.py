{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82af10e1-3764-46e3-9436-cbd3e7d33055",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np  # Додано numpy для використання NaN\n",
    "import joblib\n",
    "\n",
    "def predict():\n",
    "    file_path = r'C:\\Users\\Max Nazarenko\\Desktop\\New folder\\hidden_test.csv'\n",
    "    df = pd.read_csv(file_path)\n",
    "    df = df.drop(columns=['8'])\n",
    "    df['predictions'] = np.nan  # Замінено pd.NA на np.nan\n",
    "\n",
    "    # Загрузка моделі\n",
    "    model = joblib.load('models/model.joblib')\n",
    "\n",
    "    # Предсказання\n",
    "    predictions = model.predict(df)\n",
    "\n",
    "    # Збереження предсказань\n",
    "    pd.DataFrame(predictions, columns=['target']).to_csv('predictions/predictions.csv', index=False)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    predict()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "052f54f0-6e45-4aa6-b667-72d83b756b9e",
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
