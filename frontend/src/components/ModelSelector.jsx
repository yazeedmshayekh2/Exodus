import React, { useState, useEffect } from 'react';
import { 
  Box,
  Button,
  Menu,
  MenuItem,
  Typography,
  CircularProgress,
  Tooltip,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  IconButton,
  Alert,
  Grid,
  FormControl,
  InputLabel,
  Select,
  FormHelperText,
  Divider,
  Snackbar
} from '@mui/material';
import { 
  KeyboardArrowDown as ArrowDownIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import chatService from '../api/chatService';

const ModelSelector = ({ onModelChange, language }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentModel, setCurrentModel] = useState(null);
  const [error, setError] = useState(null);
  const [openAddDialog, setOpenAddDialog] = useState(false);
  const [addModelForm, setAddModelForm] = useState({
    repo_id: '',
    model_name: '',
    display_name: '',
    description: '',
    context_length: 4096,
    temperature: 0.7
  });
  const [formErrors, setFormErrors] = useState({});
  const [addingModel, setAddingModel] = useState(false);
  const [removingModel, setRemovingModel] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });

  const open = Boolean(anchorEl);
  
  const isRtl = language === 'ar';
  
  // Fetch available models on component mount
  useEffect(() => {
    fetchModels();
  }, []);
  
  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const availableModels = await chatService.getAvailableModels();
      setModels(availableModels);
      
      // Set current model
      const current = availableModels.find(model => model.current);
      if (current) {
        setCurrentModel(current);
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
      setError(language === 'ar'
        ? 'فشل في تحميل النماذج المتاحة'
        : 'Failed to load available models');
    } finally {
      setLoading(false);
    }
  };
  
  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleClose = () => {
    setAnchorEl(null);
  };
  
  const handleModelSelect = async (model) => {
    handleClose();
    
    if (model.id === currentModel?.id) return;
    
    try {
      setLoading(true);
      showNotification(`Switching to model: ${model.name}. This may take a moment...`, 'info');
      
      await chatService.switchModel(model.id);
      
      // Set the new current model
      setCurrentModel(model);
      
      // Update models list to reflect new current model
      setModels(prevModels => prevModels.map(m => ({
        ...m,
        current: m.id === model.id
      })));
      
      // Show success notification
      showNotification(`Successfully switched to model: ${model.name}`, 'success');
      
      // Notify parent component
      if (onModelChange) {
        onModelChange(model);
      }
    } catch (err) {
      console.error('Failed to switch model:', err);
      setError(language === 'ar'
        ? `فشل في التبديل إلى النموذج: ${model.name}`
        : `Failed to switch to model: ${model.name}`);
        
      showNotification(`Failed to switch to model: ${model.name} - ${err.response?.data?.detail || err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };
  
  const handleOpenAddDialog = () => {
    setOpenAddDialog(true);
    handleClose();
  };
  
  const handleCloseAddDialog = () => {
    setOpenAddDialog(false);
    setAddModelForm({
      repo_id: '',
      model_name: '',
      display_name: '',
      description: '',
      context_length: 4096,
      temperature: 0.7
    });
    setFormErrors({});
  };
  
  const handleAddFormChange = (event) => {
    const { name, value } = event.target;
    setAddModelForm(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error for this field if it exists
    if (formErrors[name]) {
      setFormErrors(prev => ({
        ...prev,
        [name]: undefined
      }));
    }
  };
  
  const validateForm = () => {
    const errors = {};
    
    if (!addModelForm.repo_id.trim()) {
      errors.repo_id = 'HuggingFace repository ID is required';
    }
    
    if (!addModelForm.model_name.trim()) {
      errors.model_name = 'Model name is required';
    } else if (!/^[a-zA-Z0-9_-]+$/.test(addModelForm.model_name)) {
      errors.model_name = 'Model name must contain only letters, numbers, dashes and underscores';
    }
    
    if (!addModelForm.display_name.trim()) {
      errors.display_name = 'Display name is required';
    }
    
    if (!addModelForm.description.trim()) {
      errors.description = 'Description is required';
    }
    
    if (!addModelForm.context_length || addModelForm.context_length < 1024) {
      errors.context_length = 'Context length must be at least 1024';
    }
    
    if (addModelForm.temperature < 0 || addModelForm.temperature > 1) {
      errors.temperature = 'Temperature must be between 0 and 1';
    }
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };
  
  const handleAddModel = async () => {
    if (!validateForm()) return;
    
    try {
      setAddingModel(true);
      
      // Parse numeric values
      const modelData = {
        ...addModelForm,
        context_length: parseInt(addModelForm.context_length),
        temperature: parseFloat(addModelForm.temperature)
      };
      
      // Show status notification
      showNotification(`Adding model from HuggingFace: ${modelData.repo_id}. This may take several minutes depending on model size...`, 'info');
      
      // Call API to add model
      const result = await chatService.addModelFromHuggingFace(modelData);
      
      // Close dialog and refresh models
      handleCloseAddDialog();
      
      // Show success notification
      showNotification(`Successfully added model: ${modelData.display_name}`, 'success');
      
      // Refresh the models list
      await fetchModels();
    } catch (error) {
      console.error('Failed to add model:', error);
      // Show error notification with detailed message
      const errorDetail = error.response?.data?.detail || error.message;
      showNotification(`Failed to add model: ${errorDetail}`, 'error');
    } finally {
      setAddingModel(false);
    }
  };
  
  const handleRemoveModel = async (model, event) => {
    // Stop propagation to prevent selecting the model when clicking delete
    event.stopPropagation();
    
    if (model.current) {
      showNotification('Cannot remove the currently active model', 'warning');
      return;
    }
    
    // Check if it's a built-in model
    const builtInModels = ["llama3:8b", "llama3.1:8b", "mistral:7b", "dolphin-phi3:7b", "qwen3:8b"];
    if (builtInModels.includes(model.id)) {
      showNotification('Cannot remove built-in models', 'warning');
      return;
    }
    
    try {
      setRemovingModel(true);
      showNotification(`Removing model: ${model.name}. Please wait...`, 'info');
      
      // Call API to remove model
      await chatService.removeModel(model.id);
      
      // Show success notification
      showNotification(`Successfully removed model: ${model.name}`, 'success');
      
      // Refresh models
      await fetchModels();
    } catch (error) {
      console.error('Failed to remove model:', error);
      const errorDetail = error.response?.data?.detail || error.message;
      showNotification(`Failed to remove model: ${errorDetail}`, 'error');
    } finally {
      setRemovingModel(false);
    }
  };
  
  const showNotification = (message, severity = 'info') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };
  
  const handleCloseNotification = () => {
    setNotification(prev => ({
      ...prev,
      open: false
    }));
  };
  
  // If there's only one model or none, don't show the selector
  if (models.length <= 1 && !loading && !error) {
    return null;
  }
  
  return (
    <>
      <Box sx={{ 
        mb: 2, 
        display: 'flex', 
        alignItems: 'center',
        justifyContent: 'flex-end',
        direction: isRtl ? 'rtl' : 'ltr'
      }}>
        {error && (
          <Typography 
            variant="caption" 
            color="error" 
            sx={{ mr: isRtl ? 0 : 2, ml: isRtl ? 2 : 0 }}
          >
            {error}
          </Typography>
        )}
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography 
            variant="body2" 
            color="textSecondary"
            sx={{ mr: isRtl ? 0 : 1, ml: isRtl ? 1 : 0 }}
          >
            {isRtl ? 'النموذج:' : 'Model:'}
          </Typography>
          
          <Button
            variant="outlined"
            size="small"
            onClick={handleClick}
            endIcon={<ArrowDownIcon />}
            disabled={loading || models.length === 0}
            sx={{ 
              textTransform: 'none',
              minWidth: '150px',
              justifyContent: 'space-between',
              direction: isRtl ? 'rtl' : 'ltr'
            }}
          >
            {loading ? (
              <CircularProgress size={16} thickness={5} />
            ) : (
              currentModel?.name || (isRtl ? 'اختر نموذجاً' : 'Select Model')
            )}
          </Button>
        </Box>
        
        <Menu
          anchorEl={anchorEl}
          open={open}
          onClose={handleClose}
          sx={{ maxHeight: '300px' }}
        >
          {models.map((model) => (
            <MenuItem 
              key={model.id}
              onClick={() => handleModelSelect(model)}
              selected={model.current}
              sx={{ minWidth: '250px', position: 'relative' }}
            >
              <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="body2">{model.name}</Typography>
                    {model.current && (
                      <Chip 
                        label={isRtl ? 'نشط' : 'Active'} 
                        size="small" 
                        color="primary" 
                        variant="outlined"
                        sx={{ ml: isRtl ? 0 : 1, mr: isRtl ? 1 : 0, height: '20px' }} 
                      />
                    )}
                    {model.huggingface_repo && (
                      <Chip 
                        label="HF" 
                        size="small" 
                        color="secondary" 
                        variant="outlined"
                        sx={{ ml: 1, height: '20px' }} 
                      />
                    )}
                  </Box>
                  {!model.current && (
                    <IconButton 
                      size="small" 
                      onClick={(e) => handleRemoveModel(model, e)}
                      disabled={removingModel}
                      sx={{ p: 0.5 }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  )}
                </Box>
                <Typography variant="caption" color="textSecondary">
                  {model.description}
                </Typography>
                {model.huggingface_repo && (
                  <Typography variant="caption" color="textSecondary" sx={{ fontStyle: 'italic' }}>
                    {model.huggingface_repo}
                  </Typography>
                )}
              </Box>
            </MenuItem>
          ))}
          
          <Divider sx={{ my: 1 }} />
          
          <MenuItem onClick={handleOpenAddDialog}>
            <Box sx={{ display: 'flex', alignItems: 'center', color: 'primary.main' }}>
              <AddIcon fontSize="small" sx={{ mr: 1 }} />
              <Typography variant="body2">
                {isRtl ? 'إضافة نموذج من HuggingFace' : 'Add Model from HuggingFace'}
              </Typography>
            </Box>
          </MenuItem>
        </Menu>
      </Box>
      
      {/* Add Model Dialog */}
      <Dialog 
        open={openAddDialog} 
        onClose={handleCloseAddDialog}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>
          Add Model from HuggingFace
          <IconButton
            aria-label="close"
            onClick={handleCloseAddDialog}
            sx={{
              position: 'absolute',
              right: 8,
              top: 8,
            }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="HuggingFace Repository ID"
                name="repo_id"
                value={addModelForm.repo_id}
                onChange={handleAddFormChange}
                error={!!formErrors.repo_id}
                helperText={formErrors.repo_id || "Example: 'TheBloke/Llama-2-7B-GGUF'"}
                disabled={addingModel}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Model Name (ID)"
                name="model_name"
                value={addModelForm.model_name}
                onChange={handleAddFormChange}
                error={!!formErrors.model_name}
                helperText={formErrors.model_name || "Used internally, e.g. 'llama2-7b'"}
                disabled={addingModel}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Display Name"
                name="display_name"
                value={addModelForm.display_name}
                onChange={handleAddFormChange}
                error={!!formErrors.display_name}
                helperText={formErrors.display_name || "User-friendly name"}
                disabled={addingModel}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                name="description"
                value={addModelForm.description}
                onChange={handleAddFormChange}
                error={!!formErrors.description}
                helperText={formErrors.description}
                multiline
                rows={2}
                disabled={addingModel}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Context Length"
                name="context_length"
                value={addModelForm.context_length}
                onChange={handleAddFormChange}
                error={!!formErrors.context_length}
                helperText={formErrors.context_length || "Maximum context window size"}
                disabled={addingModel}
                InputProps={{ inputProps: { min: 1024 } }}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                type="number"
                label="Temperature"
                name="temperature"
                value={addModelForm.temperature}
                onChange={handleAddFormChange}
                error={!!formErrors.temperature}
                helperText={formErrors.temperature || "Between 0.0 and 1.0"}
                disabled={addingModel}
                InputProps={{ inputProps: { min: 0, max: 1, step: 0.1 } }}
              />
            </Grid>
            <Grid item xs={12}>
              <Alert severity="info">
                Adding a model from HuggingFace will download and configure it for use with Ollama. 
                This process may take several minutes depending on your connection and the model size.
              </Alert>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={handleCloseAddDialog} 
            color="inherit"
            disabled={addingModel}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleAddModel} 
            variant="contained"
            disabled={addingModel}
            startIcon={addingModel ? <CircularProgress size={20} /> : <AddIcon />}
          >
            {addingModel ? 'Adding...' : 'Add Model'}
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Notifications */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default ModelSelector; 