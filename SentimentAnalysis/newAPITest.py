from searchtweets import ResultStream, gen_rule_payload, load_credentials,collect_results
from PreProcessor import PreProcessor
from TwitterSentimentAnalyzer import TwitterSentimentAnalyzer

import time
#Currently we are using the API can retrieve tweets on any given day (after 2006)
#However only 50 requests can be made, and a maximum of 100 tweets can be retrieved per request
premium_search_args = load_credentials("k.yaml",
                                       yaml_key="search_tweets_premium",
                                       env_overwrite=False)
#Sample search rule format
rule = gen_rule_payload("MSFT OR Microsoft", #Dollar sign not supported in premium API
                        from_date="2017-09-01",
                        to_date="2017-09-02",
                        results_per_call=10) #Do not exceed 100 per request
day30Start=['201804240000','201804230000','201804200000','201804190000','201804180000','201804170000','201804160000','201804130000','201804120000','201804110000','201804100000','201804090000','201804060000','201804050000','201804040000','201804030000','201804020000','201803290000','201803280000','201803270000','201803260000']
day30End=['201804250000','201804240000','201804210000','201804200000','201804190000','201804180000','201804170000','201804140000','201804130000','201804120000','201804110000','201804100000','201804070000','201804060000','201804050000','201804040000','201804030000','201803300000','201803290000','201803280000','201803270000']
#Get Sentiment Score
er=TwitterSentimentAnalyzer()
processor=PreProcessor()
result=[]
for i in range(0,21):
    processedTweets=[]
    rule = gen_rule_payload("MSFT OR Microsoft",
                            from_date=day30Start[i],
                            to_date=day30End[i],
                            results_per_call=80)
    tweets = collect_results(rule,
                             max_results=80,
                             result_stream_args=premium_search_args)
    for tweet in tweets[0:80]:
        r = ' '.join(word for word in processor.preProcess(tweet.all_text))
        processedTweets.append(r)
    total = 0
    for i in range(0, 2):
        tmpl = []
        for q in range(i * 40, i * 40 + 40):
            tmpl.append(processedTweets[q]);
        s = er.Evaluate(tmpl)
        for q in s:
            total = total + q
    result.append(total)
    time.sleep(1)
print(result)
