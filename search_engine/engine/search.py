from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')
BSBI_instance.load()

queries = ["dha people consuming fish and fish oil may exceed the world health organization ’ s daily safety limit for dioxins and dioxin-like substances , such as pcbs , which reduces value of fish as a dha source . in fact , fish oil may be so contaminated , it may even increase inflammatory markers , so much so that it may be not be able to counteract the adverse effects on mood caused by the arachidonic acid in fish . other industrial toxins may include endocrine-disrupting pollutants and heavy metals such as mercury . although tuna companies advertise tuna as safe and healthy for us and our children , they appear to just be employing the same techniques that chemical companies have used to try to suggest that pesticides , such as ddt , are safe and healthy . omega-3 fatty acids are therefore best obtained from fish-free sources , such as microalgae-based dha . long-chain omega-3 fatty acids are no longer found in sufficient quantities in chickens due to genetic manipulation . - omega-3 fatty acids , fish , persistent organic pollutants , algae , supplements , epa , seafood , brain health , nutrition myths , body fat , children , industrial toxins , fish oil , animal products , pregnancy - -"]
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()
