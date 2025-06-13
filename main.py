import json
from ntscraper import Nitter

# Initialize the scraper
# We can set skip_instance_check=True since we are providing our own trusted instance.
scraper = Nitter(log_level=1, skip_instance_check=False)

# Define the specific Nitter instance we want to use
# It's good practice to include the protocol (https://)
instance_url = "http://localhost:8080/"
print(f"Using specific Nitter instance: {instance_url}\n")

# --- Test Case 1: The requested user 'realdonaldtrump' ---

username = "realdonaldtrump"
num_tweets = 10

print(f"Attempting to scrape {num_tweets} tweets from user '{username}' using {instance_url}...")

# Use the get_tweets method, passing our specific instance via the 'instance' parameter
# This will likely return nothing because the account is suspended on Twitter.
# tweets = scraper.get_tweets(username, mode='user', number=num_tweets, instance=instance_url)

# Print the results for the first test case
# if tweets and tweets['tweets']:
#     print(f"\nSuccessfully scraped {len(tweets['tweets'])} tweets from '{username}'.")
#     # Loop through and print some details
#     for tweet in tweets['tweets']:
#         print(f"  - Link: {tweet['link']}")
#         print(f"    Text: {tweet['text']}")
#         print("-" * 20)
# else:
#     print(f"\nCould not retrieve tweets for user '{username}'.")
#     print(
#         "This is the expected outcome because the account is suspended on Twitter, and the Nitter instance cannot find it.")
#
# print("\n" + "=" * 50 + "\n")

# --- Test Case 2: An active user for comparison ---
#
# active_username = "X"
# print(f"Attempting to scrape {num_tweets} tweets from active user '{active_username}' using {instance_url}...")
#
# # Now, let's try with an active user to confirm the instance is working
# active_tweets = scraper.get_tweets(active_username, mode='user', number=num_tweets, instance=instance_url)
#
# if active_tweets and active_tweets['tweets']:
#     print(f"\nSuccessfully scraped {len(active_tweets['tweets'])} tweets from '{active_username}'.")
#     print("This confirms that the scraper and the instance are working correctly.")
#
#     # Print the results in a readable JSON format
#     pretty_json = json.dumps(active_tweets, indent=4)
#     print(pretty_json)
# else:
#     print(f"\nFailed to scrape tweets from '{active_username}' even with the specified instance.")
#     print(f"This might indicate an issue with '{instance_url}' or a temporary block.")



depression_tweets = scraper.get_tweets("diagnosed with depression", mode='term', number=num_tweets, instance=instance_url)
