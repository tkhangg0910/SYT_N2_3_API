Follow these steps to install the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/tkhangg0910/SYT_N2_3_API.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the project:
    ```bash
    cd BE\src
    uvicorn main:app --reload
    ```
### Setup Milvus in BE:
1. Navigate to the DB Backend directory:
    ```bash
    cd BE/src
    ```
2. Download the installation script and save it as `standalone.bat`:
    ```bash
    Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
    ```
3. Run the downloaded script to start Milvus as a Docker container:
    ```bash
    standalone.bat start
    ```
---
