# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:16:03 2024

@author: amine
"""

from pydantic import BaseModel, Extra
import forecast.utilities as util
import forecast.wrangler as wra
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from types import SimpleNamespace


class Learner(BaseModel):
    data: wra.Wrangler
    report: dict = dict(explore=None, train=None, test=None)

    class Config:
        extra = Extra.allow

    def __call__(self):
        self.report = SimpleNamespace(**self.report)

    def explore(self):
        dataset = self.data.dataset.train

        corr = util.compute_correlations(dataset, self.data.target)
        corr.sort_values(self.data.target, inplace=True)
        self.report.explore = dict(corr=corr)

        plt.figure(figsize=(16, 12))
        ax = sns.barplot(
            data=corr.reset_index(), x="index", y=self.data.target
        )

        # Set plot title and labels
        plt.title(f"Pearson correlation with `{self.data.target}`")
        plt.ylabel("Features")
        plt.xlabel("Pearson correlation coefficient")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=90)
        # plt.savefig('pearson.png')
        # Show the plot
        plt.show()

    def train(self):
        self.model = Sequential(
            [
                Dense(
                    10, input_dim=len(self.data.features), activation="relu"
                ),
                Dense(10, activation="relu"),
                Dense(1),  # Output layer with 1 output
            ]
        )

        import tensorflow as tf

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
        )

        # Train the model
        self.model.fit(
            self.data.dataset.train[self.data.features],
            self.data.dataset.train[self.data.target],
            epochs=5,
            batch_size=32,
        )

    def test(self):
        # Make predictions
        evaluation = self.model.evaluate(
            self.data.dataset.test[self.data.features],
            self.data.dataset.test[self.data.target],
        )

        self.report.test = dict(evaluation=evaluation)
