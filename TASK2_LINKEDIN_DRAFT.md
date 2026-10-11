# Task 2 LinkedIn Video and Post Draft

This draft is for the Restaurant Recommendation System task only. Record a short screen video showing the live output from `evaluate.py`; do not describe the whole internship as complete until the selected three tasks are finished.

## 45-second video plan

1. **Opening (0–5 sec):** Show the project title: “Cognifyz Machine Learning Internship — Task 2: Restaurant Recommendation”.
2. **Goal (5–12 sec):** Say: “I built a content-based system that recommends restaurants using cuisine preferences and price range.”
3. **Method (12–22 sec):** Show `recommender.py` briefly. Say: “It cleans missing cuisine values, encodes cuisine categories, and compares preference features using cosine similarity.”
4. **Live example (22–37 sec):** Run `python evaluate.py --dataset "data/Dataset .csv"`. Show the Italian and Japanese/Seafood sample recommendations and the printed cuisine-overlap and price-distance checks.
5. **Close (37–45 sec):** Show the repository link and say: “The dataset has no user-feedback labels, so I checked how well results match the requested cuisine and price preferences.”

## LinkedIn post draft

I completed Task 2 of my Cognifyz Machine Learning internship: a content-based restaurant recommendation system.

The Python project recommends restaurants based on preferred cuisines and price range. I used pandas, NumPy, and scikit-learn to preprocess the restaurant data, encode cuisine categories, and rank results with cosine similarity.

On the supplied dataset, the sample checks returned 5/5 restaurants with a requested cuisine match for both Italian/price level 2 and Japanese/Seafood/price level 3. The Japanese/Seafood results had a mean price-level difference of 0.20. These checks measure preference alignment; the dataset does not include user-feedback labels.

Project: https://github.com/anshull-rajput/cognifyz-restaurant-recommendation-system

@[Tag the official Cognifyz Technologies LinkedIn page]

#cognifyztechnologies #cognifyz #cognifyztech #MachineLearning #Python

## Before posting

- Replace the tag line by selecting the official Cognifyz Technologies page in LinkedIn.
- Confirm the dataset and results shown in the recording are readable.
- This draft covers Task 2 only; update the video and post after the other two selected internship tasks are complete if the post is intended to present the full internship.
