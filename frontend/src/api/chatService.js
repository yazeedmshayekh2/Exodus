import axios from 'axios';

// Get the API URL from environment or use a default
// Fix: Use window._env_ instead of process.env to avoid "process is not defined" error in browser
const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || (
  window.location.hostname === 'localhost' 
    ? 'http://localhost:8000/api' 
    : '/api'
);

// Create a function to test API connectivity
const testConnection = async () => {
  try {
    // First try the test endpoint
    const response = await axios.get(`${API_URL}/test`);
    console.log('API connection successful:', response.data);
    return true;
  } catch (error) {
    console.error('API connection test failed:', error);
    return false;
  }
};

// Test the connection when the service is imported
testConnection();

const chatService = {
  /**
   * Send a query to the chatbot API
   * @param {string} query - The user's query
   * @returns {Promise<Object>} - Response with the answer and language
   */
  sendQuery: async (query) => {
    try {
      const response = await axios.post(`${API_URL}/chat`, { query });
      return response.data;
    } catch (error) {
      console.error('Error sending chat query:', error);
      throw error;
    }
  },

  /**
   * Get available models that can be used with the chatbot
   * @returns {Promise<Array>} - List of available models with details
   */
  getAvailableModels: async () => {
    try {
      const response = await axios.get(`${API_URL}/models`);
      return response.data.models;
    } catch (error) {
      console.error('Error getting available models:', error);
      throw error;
    }
  },

  /**
   * Switch to a different model
   * @param {string} modelId - The ID of the model to switch to
   * @returns {Promise<Object>} - Result of the model switch operation
   */
  switchModel: async (modelId) => {
    try {
      const response = await axios.post(`${API_URL}/models/switch`, { model_id: modelId });
      return response.data;
    } catch (error) {
      console.error('Error switching model:', error);
      throw error;
    }
  },

  /**
   * Add a new model from HuggingFace
   * @param {Object} modelData - Model details
   * @param {string} modelData.repo_id - HuggingFace repository ID
   * @param {string} modelData.model_name - Name to use for the model in Ollama
   * @param {string} modelData.display_name - Human-readable name for the model
   * @param {string} modelData.description - Description of the model
   * @param {number} [modelData.context_length=4096] - Context window size
   * @param {number} [modelData.temperature=0.7] - Default temperature
   * @returns {Promise<Object>} - Result of the add operation
   */
  addModelFromHuggingFace: async (modelData) => {
    try {
      const response = await axios.post(`${API_URL}/models/add`, modelData);
      return response.data;
    } catch (error) {
      console.error('Error adding model from HuggingFace:', error);
      throw error;
    }
  },

  /**
   * Remove a model
   * @param {string} modelId - The ID of the model to remove
   * @returns {Promise<Object>} - Result of the remove operation
   */
  removeModel: async (modelId) => {
    try {
      const response = await axios.post(`${API_URL}/models/remove`, { model_id: modelId });
      return response.data;
    } catch (error) {
      console.error('Error removing model:', error);
      throw error;
    }
  },

  /**
   * Get debug information about FAQs
   * @returns {Promise<Object>} - FAQ count and sample FAQs
   */
  getDebugFaqs: async () => {
    try {
      const response = await axios.get(`${API_URL}/debug/faqs`);
      return response.data;
    } catch (error) {
      console.error('Error getting debug FAQs:', error);
      throw error;
    }
  },

  /**
   * Test a query against the FAQ matching system
   * @param {string} query - The test query
   * @returns {Promise<Object>} - Debug information about the match
   */
  testFaqMatch: async (query) => {
    try {
      const response = await axios.get(`${API_URL}/debug/faq/${encodeURIComponent(query)}`);
      return response.data;
    } catch (error) {
      console.error('Error testing FAQ match:', error);
      throw error;
    }
  },

  /**
   * Get server information
   * @returns {Promise<Object>} - Server details and configuration
   */
  getServerInfo: async () => {
    try {
      const response = await axios.get(`${API_URL}/debug/server-info`);
      return response.data;
    } catch (error) {
      console.error('Error getting server info:', error);
      throw error;
    }
  },
  
  /**
   * Test the API connection
   * @returns {Promise<boolean>} - Whether the connection was successful
   */
  testConnection
};

export default chatService; 