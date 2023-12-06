# ERRORS
I got during run K-finBERT

0. Environment download error. 
    In Windows11, I got a version conflict error between python and torch, so install recommended torch version instead of 1.1.0. And in Linux (Ubuntu), I got conda error below
    ```bash
    WARNING conda.models.version:get_matcher(546)...
    ```
    *Solution*
    ```bash
    # environment.yml
    - transformers=4.1.1
    - pip # Add this line
    - pip:
        - joblib==0.13.2
    ```

1. Attribute error
    It might happens when there is a version conflict between Numpy and Pandas (not always)
    ```bash
    AttributeError:type object 'object' has no attribute 'dtype'
    ```
    *Solution*
    ```bash
    pip install --upgrade numpy
    pip install --upgrade pandas
    ```

1. Symbolic Link error in Conda
    This error might happen due to Conda version [Reference](https://github.com/conda/conda/issues/9957)
    ```bash
    libffi.so.7: cannot open shared object file: No such file or directory
    ```
    *Solutions*
    ```bash
    # 1. Link symbols (sub-optimal)
    ln -s libffi.so.7.1.0 libffi.so.6
    # 2. Redownload your conda (best)

    # 3. Replace pip install with conda install (simple and easy)
    conda install scikit-learn==0.21.2
    ```

2. SPM Reference error
    ```bash
    UnboundLocalError: local variable 'spm' referenced before assignment
    ```
    *Solution*
    ```bash
    # Install sentencepiece first
    pip install sentencepiece
    ```

3. GPU error
    ```bash
    cublas runtime error : the GPU program failed to execute at /pytorch/aten/src/THC/THCBlas.cu:450
    ```
    *Solution*
    ```bash
    # Upgrade torch version
    pip install --upgrade torch
    ```

4. NLTK error
    ```bash
    LookupError: 
        Resource punkt not found.
        Please use the NLTK Downloader to obtain the resource:
        ...
    ```
    *Solution*
    ```python
    import nltk
    nltk.download('punkt')
    ```