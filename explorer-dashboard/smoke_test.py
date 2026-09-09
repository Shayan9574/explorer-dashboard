from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=120)
at.run()
if at.exception:
    for e in at.exception:
        print("EXCEPTION:", e.value)
        print(e.stack_trace if hasattr(e,'stack_trace') else '')
else:
    print("No exceptions. Tabs:", len(at.tabs), "| Metrics:", len(at.metric), "| Dataframes:", len(at.dataframe))
    # sanity: verify computed numbers match the verified analysis
    import sys; sys.path.insert(0,'.')
