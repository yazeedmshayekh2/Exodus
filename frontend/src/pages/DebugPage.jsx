import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  TextField, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  CardHeader,
  Divider,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress
} from '@mui/material';
import { 
  BugReport as BugIcon,
  ExpandMore as ExpandMoreIcon,
  Search as SearchIcon, 
  Info as InfoIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import chatService from '../api/chatService';

const MotionPaper = motion(Paper);
const MotionCard = motion(Card);

const DebugPage = () => {
  const [serverInfo, setServerInfo] = useState(null);
  const [faqInfo, setFaqInfo] = useState(null);
  const [testQuery, setTestQuery] = useState('');
  const [testResult, setTestResult] = useState(null);
  const [loading, setLoading] = useState({
    serverInfo: false,
    faqInfo: false,
    testQuery: false
  });
  const [error, setError] = useState({
    serverInfo: null,
    faqInfo: null,
    testQuery: null
  });

  useEffect(() => {
    fetchServerInfo();
    fetchFaqInfo();
  }, []);

  const fetchServerInfo = async () => {
    try {
      setLoading(prev => ({ ...prev, serverInfo: true }));
      setError(prev => ({ ...prev, serverInfo: null }));
      const info = await chatService.getServerInfo();
      setServerInfo(info);
    } catch (err) {
      setError(prev => ({ 
        ...prev, 
        serverInfo: err.response?.data?.detail || err.message 
      }));
    } finally {
      setLoading(prev => ({ ...prev, serverInfo: false }));
    }
  };

  const fetchFaqInfo = async () => {
    try {
      setLoading(prev => ({ ...prev, faqInfo: true }));
      setError(prev => ({ ...prev, faqInfo: null }));
      const info = await chatService.getDebugFaqs();
      setFaqInfo(info);
    } catch (err) {
      setError(prev => ({ 
        ...prev, 
        faqInfo: err.response?.data?.detail || err.message 
      }));
    } finally {
      setLoading(prev => ({ ...prev, faqInfo: false }));
    }
  };

  const handleTestQuery = async () => {
    if (!testQuery.trim()) return;
    
    try {
      setLoading(prev => ({ ...prev, testQuery: true }));
      setError(prev => ({ ...prev, testQuery: null }));
      setTestResult(null);
      const result = await chatService.testFaqMatch(testQuery);
      setTestResult(result);
    } catch (err) {
      setError(prev => ({ 
        ...prev, 
        testQuery: err.response?.data?.detail || err.message 
      }));
    } finally {
      setLoading(prev => ({ ...prev, testQuery: false }));
    }
  };

  return (
    <Box>
      <Typography variant="h3" component="h1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <BugIcon fontSize="large" /> Debug Interface
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        This page provides tools to test and debug the FAQ chatbot API.
      </Typography>

      <Grid container spacing={4}>
        {/* Test Query Section */}
        <Grid item xs={12} md={6}>
          <MotionPaper
            elevation={3}
            sx={{ p: 3, height: '100%' }}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <SearchIcon color="primary" /> Test Query Matching
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Test how the system matches queries to FAQs and view similarity scores.
            </Typography>
            
            <Box sx={{ mt: 2, mb: 3 }}>
              <TextField
                fullWidth
                label="Enter a test query"
                variant="outlined"
                value={testQuery}
                onChange={(e) => setTestQuery(e.target.value)}
                sx={{ mb: 2 }}
              />
              <Button 
                variant="contained" 
                startIcon={<SearchIcon />} 
                onClick={handleTestQuery}
                disabled={loading.testQuery || !testQuery.trim()}
              >
                {loading.testQuery ? <CircularProgress size={24} /> : 'Test Query'}
              </Button>
            </Box>
            
            {error.testQuery && (
              <Alert severity="error" sx={{ mt: 2 }}>{error.testQuery}</Alert>
            )}
            
            {testResult && (
              <MotionCard 
                variant="outlined" 
                sx={{ mt: 3 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <CardHeader 
                  title="Test Results" 
                  subheader={`Query: "${testResult.query}"`} 
                />
                <Divider />
                <CardContent>
                  <Typography variant="body2" paragraph>
                    Language: <strong>{testResult.language}</strong>
                  </Typography>
                  <Typography variant="body2" paragraph>
                    Similarity Score: <strong>{testResult.similarity_score.toFixed(4)}</strong>
                  </Typography>
                  <Typography variant="body2" paragraph>
                    Threshold: <strong>{testResult.threshold.toFixed(4)}</strong>
                  </Typography>
                  <Typography variant="body2" paragraph>
                    Above Threshold: <strong>{testResult.exceeds_threshold ? 'Yes' : 'No'}</strong>
                  </Typography>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="subtitle1" gutterBottom>
                    Best Match:
                  </Typography>
                  {testResult.best_match ? (
                    <>
                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                        Question (EN):
                      </Typography>
                      <Typography variant="body2" paragraph sx={{ ml: 2 }}>
                        {testResult.best_match.question_en}
                      </Typography>
                      
                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                        Question (AR):
                      </Typography>
                      <Typography variant="body2" paragraph sx={{ ml: 2 }}>
                        {testResult.best_match.question_ar}
                      </Typography>
                      
                      <Accordion>
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                          <Typography>View Answer</Typography>
                        </AccordionSummary>
                        <AccordionDetails>
                          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            Answer (EN):
                          </Typography>
                          <Typography variant="body2" paragraph sx={{ ml: 2 }}>
                            {testResult.best_match.answer_en}
                          </Typography>
                          
                          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            Answer (AR):
                          </Typography>
                          <Typography variant="body2" paragraph sx={{ ml: 2, direction: 'rtl', textAlign: 'right' }}>
                            {testResult.best_match.answer_ar}
                          </Typography>
                        </AccordionDetails>
                      </Accordion>
                    </>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No match found.
                    </Typography>
                  )}
                </CardContent>
              </MotionCard>
            )}
          </MotionPaper>
        </Grid>
        
        {/* FAQ Information Section */}
        <Grid item xs={12} md={6}>
          <MotionPaper
            elevation={3}
            sx={{ p: 3 }}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <InfoIcon color="primary" /> System Information
            </Typography>
            
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                FAQ Database
              </Typography>
              
              {loading.faqInfo ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : error.faqInfo ? (
                <Alert severity="error" sx={{ mt: 2 }}>{error.faqInfo}</Alert>
              ) : faqInfo ? (
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="body1" paragraph>
                      Total FAQs: <strong>{faqInfo.faq_count}</strong>
                    </Typography>
                    
                    {faqInfo.sample_faqs && faqInfo.sample_faqs.length > 0 && (
                      <>
                        <Typography variant="subtitle1" gutterBottom>
                          Sample FAQs:
                        </Typography>
                        {faqInfo.sample_faqs.map((faq, index) => (
                          <Accordion key={index} sx={{ mb: 1 }}>
                            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                              <Typography variant="body2">
                                {index + 1}. {faq.question_en}
                              </Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                Question (AR):
                              </Typography>
                              <Typography variant="body2" paragraph sx={{ direction: 'rtl', textAlign: 'right' }}>
                                {faq.question_ar}
                              </Typography>
                            </AccordionDetails>
                          </Accordion>
                        ))}
                      </>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No FAQ information available.
                </Typography>
              )}
            </Box>

            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Server Status
              </Typography>
              
              {loading.serverInfo ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
                  <CircularProgress />
                </Box>
              ) : error.serverInfo ? (
                <Alert severity="error" sx={{ mt: 2 }}>{error.serverInfo}</Alert>
              ) : serverInfo ? (
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle1" gutterBottom>
                      Server:
                    </Typography>
                    <Typography variant="body2" paragraph>
                      Local URL: <strong>{serverInfo.server.localhost_url}</strong>
                    </Typography>
                    <Typography variant="body2" paragraph>
                      LAN URL: <strong>{serverInfo.server.local_lan_url}</strong>
                    </Typography>
                    <Typography variant="body2" paragraph>
                      SSL Enabled: <strong>{serverInfo.server.ssl_enabled ? 'Yes' : 'No'}</strong>
                    </Typography>
                    
                    {serverInfo.server.ngrok_url && (
                      <Typography variant="body2" paragraph>
                        Ngrok URL: <strong>{serverInfo.server.ngrok_url}</strong>
                      </Typography>
                    )}
                    
                    <Typography variant="subtitle1" gutterBottom>
                      Environment:
                    </Typography>
                    <Typography variant="body2" paragraph>
                      Host: <strong>{serverInfo.environment.host}</strong>
                    </Typography>
                    <Typography variant="body2" paragraph>
                      Port: <strong>{serverInfo.environment.port}</strong>
                    </Typography>
                  </CardContent>
                </Card>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No server information available.
                </Typography>
              )}
            </Box>
          </MotionPaper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DebugPage;