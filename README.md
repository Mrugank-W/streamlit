# Privacy-Preserving Machine Learning using TenSEAL

This project demonstrates a **privacy-preserving machine learning pipeline** where inference is performed directly on **encrypted data** using **homomorphic encryption**.  
Sensitive input data remains encrypted throughout computation, ensuring confidentiality during machine learning inference.

---

## Overview

Conventional machine learning systems require access to plaintext data, which poses privacy and security risks in sensitive domains.  
This project addresses that issue by integrating **homomorphic encryption** with machine learning, enabling computations on encrypted inputs without revealing the original data.

The implementation uses **TenSEAL**, a Python library built on Microsoft SEAL, to perform secure encrypted inference.

---

## Key Concepts

- Homomorphic Encryption  
- Privacy-Preserving Machine Learning  
- Encrypted Inference  
- Secure Computation  
- Accuracy vs Performance Trade-offs  

---

## Tech Stack

- Language: Python  
- Security Library: TenSEAL  
- ML Libraries: NumPy, scikit-learn  
- Encryption Scheme: CKKS  
- Environment: Jupyter Notebook / Google Colab  

---

## Workflow

1. Train a machine learning model on plaintext data  
2. Encrypt input features using the CKKS scheme  
3. Perform inference on encrypted inputs  
4. Decrypt the encrypted output to obtain predictions  
5. Ensure no plaintext data is exposed during computation  

---

## Security Highlights

- Input data remains encrypted at all stages  
- Model never accesses raw user data  
- Suitable for data-sensitive domains such as healthcare and finance  
- Prevents data leakage during ML inference  

---

## Performance Considerations

- Encrypted computation introduces computational overhead  
- Slight numerical approximation due to CKKS encryption  
- Demonstrates trade-offs between privacy, accuracy, and latency  

---

## Installation

```bash
pip install tenseal numpy scikit-learn
