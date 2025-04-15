### Load OpenAI API 

Store your OpenAI API in by using 'export OPENAI_API_KEY="your-api-key-here"' or set the OpenAI API key as an environment variable in your system log. 

### Install TRIX Package

```bash
pip install -e . 
```

We created a jupyter notebook in /tests/run_eval.ipynb to reproduce the results, including execute TRIX to extract structured data from our benchmark, and evaluate the results to return precision and recall. 

### Execute the following command line to run the main code. 
In /tests/run_eval.ipynb, first execute 
```bash
trix.key_prediction(pdf_path)
```
by passing document path, then perform 
```bash
trix.template_based_data_extraction(pdf_path, out_path)
``` 
by passing document path and the path to store the results. 

### Execute the following command line to evaluate the results. 

```bash
trix.eval(method)
```
by passing the method as "TRIX". See /tests/run_eval.ipynb for how to invoke the libraries for data extraction and evaluation. 


The raw data is stored in data/raw, and the ground truth data is stored in data/truths. The results returned by TWIX is stored in out/. 



