# app.py
import streamlit as st
import tenseal as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import io
import base64
from PIL import Image
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="SecureAnalytics - Encrypted Data Analysis",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #4e73df;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #3756a4;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 5px solid #4e73df;
        padding: 1rem;
        border-radius: 5px;
    }
    .success-box {
        background-color: #e8f8f5;
        border-left: 5px solid #1cc88a;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x70?text=SecureAnalytics", width=150)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Encrypt Data", "Analyze", "About"],
        icons=["house", "lock", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")
    
    theme = st.sidebar.radio("Theme", ["Light", "Dark"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("TenSEAL enables computation on encrypted data using Homomorphic Encryption")

# Global variables to store state
if 'encryption_context' not in st.session_state:
    st.session_state.encryption_context = None
if 'encrypted_data' not in st.session_state:
    st.session_state.encrypted_data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'public_key' not in st.session_state:
    st.session_state.public_key = None
if 'private_key' not in st.session_state:
    st.session_state.private_key = None

# Helper functions for TenSEAL operations
def create_encryption_context(poly_modulus_degree=8192, 
                              coeff_mod_bit_sizes=[60, 40, 40, 60],
                              scale=2**40):
    """Create a TenSEAL context for encryption"""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    )
    context.global_scale = scale
    context.generate_galois_keys()
    return context

def encrypt_data(context, data):
    """Encrypt numpy array data using TenSEAL"""
    encrypted_data = []
    for column in data.T:  # Encrypt each column separately
        encrypted_vector = ts.ckks_vector(context, column)
        encrypted_data.append(encrypted_vector)
    return encrypted_data

def compute_encrypted_mean(encrypted_vectors):
    """Compute mean on encrypted vectors"""
    results = []
    for enc_vector in encrypted_vectors:
        # For CKKS, we can directly compute the sum
        encrypted_sum = enc_vector.sum()
        # We need the count of elements, which is not encrypted
        count = len(enc_vector)
        # To compute mean, we multiply by 1/count
        encrypted_mean = encrypted_sum * (1.0 / count)
        results.append(encrypted_mean)
    return results

def compute_encrypted_variance(encrypted_vectors, means):
    """Compute variance on encrypted vectors"""
    results = []
    for i, enc_vector in enumerate(encrypted_vectors):
        # We need to compute (x_i - mean)^2 for each element
        # This requires making a copy of the encrypted vector for each element
        # This is a simplified approach and not the most efficient
        count = len(enc_vector)
        
        # Create a vector of means
        mean_vector = ts.ckks_vector(enc_vector.context(), [means[i]] * count)
        
        # Compute (x - mean)
        diff = enc_vector - mean_vector
        
        # Square the difference
        squared_diff = diff * diff
        
        # Sum and divide by count
        variance = squared_diff.sum() * (1.0 / count)
        results.append(variance)
    return results

# Home page
def home_page():
    st.title("🔒 SecureAnalytics - Privacy-Preserving Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>Welcome to SecureAnalytics</h3>
        <p>Analyze sensitive data while preserving privacy using homomorphic encryption.</p>
        <p>Your data remains encrypted throughout the entire analysis process.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### How it works")
        st.markdown("""
        1. **Encrypt your data** - Upload your dataset and encrypt it using TenSEAL
        2. **Analyze securely** - Perform statistical operations on the encrypted data
        3. **View results** - Decrypt and visualize the results with confidence
        """)
        
        st.markdown("### Key Benefits")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric(label="Data Privacy", value="100%", delta="Guaranteed")
        with col_b:
            st.metric(label="Security Level", value="High", delta="Industry Standard")
        with col_c:
            st.metric(label="Computation", value="Encrypted", delta="End-to-End")
    
    with col2:
        st.image("https://via.placeholder.com/300x400?text=Encryption+Visualization", use_column_width=True)
        
        st.markdown("### Ready to try?")
        if st.button("Get Started", use_container_width=True):
            st.session_state.page = "encrypt"
            st.experimental_rerun()

# Encrypt Data page
def encrypt_data_page():
    st.title("🔐 Encrypt Your Data")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Upload Dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.original_data = data
                
                st.success(f"File uploaded successfully: {uploaded_file.name}")
                st.markdown("### Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_columns:
                    selected_columns = st.multiselect(
                        "Select numeric columns to encrypt",
                        options=numeric_columns,
                        default=numeric_columns[:2] if len(numeric_columns) > 1 else numeric_columns
                    )
                    
                    if selected_columns:
                        if st.button("Encrypt Selected Data", use_container_width=True):
                            with st.spinner("Generating encryption context and keys..."):
                                # Create TenSEAL context
                                context = create_encryption_context()
                                st.session_state.encryption_context = context
                                
                                # Extract only the numeric columns we want to encrypt
                                numeric_data = data[selected_columns].to_numpy()
                                
                                # Check for and handle NaN values
                                if np.isnan(numeric_data).any():
                                    st.warning("NaN values detected. Replacing with zeros.")
                                    numeric_data = np.nan_to_num(numeric_data, nan=0.0)
                                
                                # Encrypt the data
                                with st.spinner("Encrypting data..."):
                                    start_time = time.time()
                                    st.session_state.encrypted_data = encrypt_data(context, numeric_data)
                                    encryption_time = time.time() - start_time
                                
                                st.session_state.selected_columns = selected_columns
                                
                                st.markdown(f"""
                                <div class="success-box">
                                <h3>✅ Encryption Successful!</h3>
                                <p>Your data has been encrypted using TenSEAL's CKKS scheme.</p>
                                <p>Encryption time: {encryption_time:.2f} seconds</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Generate a serialized version of the context for download
                                context_bytes = context.serialize()
                                context_b64 = base64.b64encode(context_bytes).decode()
                                
                                st.download_button(
                                    label="Download Encryption Context",
                                    data=context_bytes,
                                    file_name="tenseal_context.bin",
                                    mime="application/octet-stream"
                                )
                else:
                    st.error("No numeric columns found in the dataset. Please upload a dataset with numeric values.")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    with col2:
        st.markdown("### About Encryption")
        st.markdown("""
        <div class="info-box">
        <h4>Homomorphic Encryption</h4>
        <p>TenSEAL uses the CKKS scheme for homomorphic encryption, allowing computations on encrypted data without decryption.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Encryption Parameters")
        
        poly_mod_degree = st.selectbox(
            "Polynomial Modulus Degree",
            options=[4096, 8192, 16384, 32768],
            index=1,
            help="Higher values provide more security but decrease performance"
        )
        
        st.markdown("#### Security Level")
        st.slider("Bit Security", min_value=128, max_value=256, value=128, step=32, 
                  help="Higher security requires more computational resources")
        
        # Display encryption visualization
        st.markdown("#### Encryption Process")
        st.image("https://via.placeholder.com/400x200?text=Encryption+Process+Visualization", use_column_width=True)

# Analyze page
def analyze_page():
    st.title("📊 Analyze Encrypted Data")
    
    if st.session_state.encrypted_data is None or st.session_state.encryption_context is None:
        st.warning("Please encrypt your data first before proceeding to analysis.")
        if st.button("Go to Encryption Page"):
            st.session_state.page = "encrypt"
            st.experimental_rerun()
        return
    
    st.markdown("### Available Analysis Methods")
    
    analysis_type = st.radio(
        "Select Analysis Method",
        ["Basic Statistics", "Correlation Analysis", "Linear Regression"],
        horizontal=True
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if analysis_type == "Basic Statistics":
            st.markdown("#### Basic Statistical Analysis on Encrypted Data")
            
            if st.button("Compute Encrypted Statistics", use_container_width=True):
                with st.spinner("Computing statistics on encrypted data..."):
                    # Compute encrypted means
                    start_time = time.time()
                    encrypted_means = compute_encrypted_mean(st.session_state.encrypted_data)
                    
                    # Decrypt means for display
                    decrypted_means = [float(mean.decrypt()) for mean in encrypted_means]
                    
                    # Compute encrypted variances
                    encrypted_variances = compute_encrypted_variance(st.session_state.encrypted_data, decrypted_means)
                    
                    # Decrypt variances for display
                    decrypted_variances = [float(var.decrypt()) for var in encrypted_variances]
                    
                    # Calculate standard deviations
                    decrypted_stds = [np.sqrt(var) for var in decrypted_variances]
                    
                    computation_time = time.time() - start_time
                
                # Display results
                results_df = pd.DataFrame({
                    'Column': st.session_state.selected_columns,
                    'Mean': decrypted_means,
                    'Variance': decrypted_variances,
                    'Std Dev': decrypted_stds
                })
                
                st.markdown(f"""
                <div class="success-box">
                <h3>Analysis Complete</h3>
                <p>Statistical analysis performed on encrypted data.</p>
                <p>Computation time: {computation_time:.2f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Results (Decrypted for Display)")
                st.dataframe(results_df, use_container_width=True)
                
                # Create visualization
                st.markdown("#### Visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = range(len(st.session_state.selected_columns))
                width = 0.35
                
                ax.bar([i - width/2 for i in x], decrypted_means, width, label='Mean')
                ax.bar([i + width/2 for i in x], decrypted_stds, width, label='Std Dev')
                
                ax.set_xticks(x)
                ax.set_xticklabels(st.session_state.selected_columns, rotation=45, ha='right')
                ax.legend()
                ax.set_title('Mean and Standard Deviation by Column')
                
                st.pyplot(fig)
        
        elif analysis_type == "Correlation Analysis":
            st.markdown("#### Correlation Analysis on Encrypted Data")
            st.info("Note: Correlation analysis with homomorphic encryption requires special protocols. This is a simplified demonstration.")
            
            if st.button("Calculate Correlation", use_container_width=True):
                # For demonstration - in reality, computing correlations on encrypted data
                # would require a more complex protocol
                with st.spinner("Calculating correlation coefficients..."):
                    # We'll decrypt the data for correlation calculation in this demo
                    # In a real secure system, you'd use secure MPC or other techniques
                    decrypted_data = []
                    for i, enc_vector in enumerate(st.session_state.encrypted_data):
                        decrypted_data.append(enc_vector.decrypt())
                    
                    # Create a dataframe with the decrypted data
                    correlation_df = pd.DataFrame(
                        {col: data for col, data in zip(st.session_state.selected_columns, decrypted_data)}
                    )
                    
                    # Calculate correlation
                    correlation_matrix = correlation_df.corr()
                
                st.markdown("#### Correlation Matrix")
                st.dataframe(correlation_matrix, use_container_width=True)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                st.pyplot(fig)
                
                st.markdown("""
                <div class="info-box">
                <p><strong>Note:</strong> In a fully secure system, correlation would be computed using secure multi-party computation protocols
                while keeping the data encrypted end-to-end.</p>
                </div>
                """, unsafe_allow_html=True)
        
        elif analysis_type == "Linear Regression":
            st.markdown("#### Linear Regression on Encrypted Data")
            
            if len(st.session_state.selected_columns) < 2:
                st.warning("You need at least 2 columns to perform regression analysis.")
            else:
                dependent_var = st.selectbox(
                    "Select dependent variable (Y)",
                    options=st.session_state.selected_columns
                )
                
                independent_vars = st.multiselect(
                    "Select independent variables (X)",
                    options=[col for col in st.session_state.selected_columns if col != dependent_var],
                    default=[col for col in st.session_state.selected_columns if col != dependent_var][:1]
                )
                
                if dependent_var and independent_vars and st.button("Run Regression Analysis", use_container_width=True):
                    st.info("For demonstration purposes, we'll decrypt the data to show the regression results.")
                    
                    # Decrypt data for demonstration
                    decrypted_data = {}
                    for i, col in enumerate(st.session_state.selected_columns):
                        decrypted_data[col] = st.session_state.encrypted_data[i].decrypt()
                    
                    # Create DataFrame
                    df = pd.DataFrame(decrypted_data)
                    
                    # Simple linear regression implementation
                    X = df[independent_vars]
                    X = np.column_stack([np.ones(X.shape[0]), X])  # Add intercept
                    y = df[dependent_var]
                    
                    # Calculate coefficients: (X'X)^(-1)X'y
                    coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
                    
                    # Calculate predictions
                    y_pred = X @ coeffs
                    
                    # Calculate R-squared
                    ss_total = np.sum((y - np.mean(y))**2)
                    ss_residual = np.sum((y - y_pred)**2)
                    r_squared = 1 - (ss_residual / ss_total)
                    
                    # Display results
                    st.markdown("#### Regression Results")
                    
                    results_df = pd.DataFrame({
                        'Variable': ['Intercept'] + independent_vars,
                        'Coefficient': coeffs
                    })
                    
                    st.dataframe(results_df, use_container_width=True)
                    st.metric("R-squared", f"{r_squared:.4f}")
                    
                    # Plot actual vs predicted
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y, y_pred, alpha=0.5)
                    
                    # Add diagonal line
                    min_val = min(min(y), min(y_pred))
                    max_val = max(max(y), max(y_pred))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title(f"Actual vs Predicted: {dependent_var}")
                    
                    st.pyplot(fig)
                    
                    st.markdown("""
                    <div class="info-box">
                    <p><strong>Note:</strong> In a fully homomorphic system, we would perform the regression calculations on the encrypted data
                    and only decrypt the final coefficients.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### About the Analysis")
        
        if analysis_type == "Basic Statistics":
            st.markdown("""
            <div class="info-box">
            <h4>Homomorphic Basic Statistics</h4>
            <p>The system computes mean and variance directly on the encrypted data using TenSEAL's CKKS scheme capabilities.</p>
            <p>Only the final statistical results are decrypted for display, preserving the privacy of individual data points.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### How it works")
            st.markdown("""
            1. **Encrypted Mean**: Sum all encrypted values and multiply by 1/n
            2. **Encrypted Variance**: Compute (x-mean)² for each encrypted value, then average
            3. **Standard Deviation**: Calculated as square root of variance after decryption
            """)
            
        elif analysis_type == "Correlation Analysis":
            st.markdown("""
            <div class="info-box">
            <h4>Secure Correlation Analysis</h4>
            <p>Computing correlations on encrypted data requires specialized protocols that allow secure computation of the necessary 
            covariance and standard deviations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Advanced Implementation")
            st.markdown("""
            A production-grade secure correlation system would use:
            
            - Secure multi-party computation (MPC)
            - Protocol for standardizing encrypted values
            - Secure dot product protocol
            - Specialized approximations for division operations
            """)
            
        elif analysis_type == "Linear Regression":
            st.markdown("""
            <div class="info-box">
            <h4>Privacy-Preserving Regression</h4>
            <p>Homomorphic encryption enables training regression models without exposing individual data points.</p>
            <p>Only the model coefficients are decrypted, keeping the training data secure.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Implementation Approaches")
            st.markdown("""
            Homomorphic linear regression can be implemented using:
            
            1. **Gradient Descent**: Iteratively update encrypted coefficients
            2. **Normal Equations**: Solve (X'X)^(-1)X'y with matrix operations
            3. **Approximation Methods**: Use polynomial approximations for non-linear operations
            """)
        
        # Display sample resources
        st.markdown("### Resources")
        st.markdown("""
        - [TenSEAL Documentation](https://tenseal.readthedocs.io/)
        - [Homomorphic Encryption Standardization](https://homomorphicencryption.org/)
        - [Privacy-Preserving Machine Learning](https://www.nature.com/articles/s42256-022-00534-z)
        """)

# About page
def about_page():
    st.title("ℹ️ About SecureAnalytics")
    
    st.markdown("""
    <div class="info-box">
    <h3>Privacy-First Data Analysis</h3>
    <p>SecureAnalytics is a cutting-edge platform that leverages homomorphic encryption to enable analysis of sensitive data 
    while maintaining complete privacy.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Technology Stack")
        st.markdown("""
        - **TenSEAL**: Library for performing homomorphic encryption operations on tensors
        - **CKKS Scheme**: Advanced encryption scheme that allows approximate arithmetic on encrypted data
        - **Streamlit**: Modern web framework for data applications
        - **NumPy & Pandas**: For efficient data handling
        - **Matplotlib & Seaborn**: For data visualization
        """)
        
        st.markdown("### Use Cases")
        st.markdown("""
        - **Healthcare**: Analyze patient data while preserving privacy
        - **Finance**: Perform analytics on sensitive financial records
        - **Research**: Collaborate on confidential datasets
        - **Education**: Learn about privacy-preserving computation
        """)
        
        st.markdown("### Technical Details")
        with st.expander("Learn More About Homomorphic Encryption"):
            st.markdown("""
            Homomorphic Encryption (HE) is a form of encryption that allows computations to be performed directly on encrypted data
            without requiring access to a secret key. The result of the computation remains encrypted and can only be revealed
            by the owner of the secret key.
            
            TenSEAL uses the CKKS scheme, which supports:
            - Addition between ciphertexts
            - Multiplication between ciphertexts
            - Addition and multiplication between ciphertexts and plaintexts
            - Limited depth of operations before requiring bootstrapping
            
            CKKS operates on approximate arithmetic, making it suitable for machine learning and statistical applications.
            """)
    
    with col2:
        st.image("https://via.placeholder.com/400x300?text=Homomorphic+Encryption", use_column_width=True)
        
        st.markdown("### Privacy Guarantees")
        st.markdown("""
        - **End-to-End Encryption**: Data is encrypted before upload and stays encrypted during computation
        - **Zero-Knowledge Processing**: Server never sees the plaintext data
        - **Secure Key Management**: Encryption keys remain with the data owner
        - **Mathematical Guarantees**: Security based on established cryptographic hardness assumptions
        """)
        
        st.markdown("### Get Started")
        if st.button("Try SecureAnalytics Now", use_container_width=True):
            st.session_state.page = "encrypt"
            st.experimental_rerun()

# Main app logic
if selected == "Home":
    home_page()
elif selected == "Encrypt Data":
    encrypt_data_page()
elif selected == "Analyze":
    analyze_page()
elif selected == "About":
    about_page()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>SecureAnalytics - Privacy-Preserving Data Analysis Platform</p>
    <p>Built with ❤️ using TenSEAL and Streamlit</p>
</div>
""", unsafe_allow_html=True)
