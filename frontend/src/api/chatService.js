import axios from 'axios';

// Try to access the API at both HTTP and HTTPS endpoints
const API_URL = 'http://localhost:8080/api';

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