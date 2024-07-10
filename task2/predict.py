{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9da6912-d909-42dc-a443-17b9b82514b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import os\n",
    "\n",
    "def predict():\n",
    "    current_dir = os.path.dirname(__file__)\n",
    "    \n",
    "    file_path = os.path.join(current_dir, 'hidden_test.csv')\n",
    "    \n",
    "    df = pd.read_csv(file_path)\n",
    "    df = df.drop(columns=['8'])\n",
    "    df['predictions'] = np.nan\n",
    "\n",
    "    model_path = os.path.join(current_dir, 'models', 'model.joblib')\n",
    "    model = joblib.load(model_path)\n",
    "\n",
    "    predictions = model.predict(df)\n",
    "\n",
    "    output_dir = os.path.join(current_dir, 'predictions')\n",
    "    os.makedirs(output_dir, exist_ok=True)\n",
    "    predictions_file = os.path.join(output_dir, 'predictions.csv')\n",
    "    pd.DataFrame(predictions, columns=['target']).to_csv(predictions_file, index=False)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    predict()"
   ]
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
