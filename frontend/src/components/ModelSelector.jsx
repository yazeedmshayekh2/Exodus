import React, { useState, useEffect } from 'react';
import { 
  Box,
  Button,
  Menu,
  MenuItem,
  Typography,
  CircularProgress,
  Tooltip,
  Chip
} from '@mui/material';
import { KeyboardArrowDown as ArrowDownIcon } from '@mui/icons-material';
import chatService from '../api/chatService';

const ModelSelector = ({ onModelChange, language }) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentModel, setCurrentModel] = useState(null);
  const [error, setError] = useState(null);
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
      await chatService.switchModel(model.id);
      setCurrentModel(model);
      
      // Update models list to reflect new current model
      setModels(models.map(m => ({
        ...m,
        current: m.id === model.id
      })));
      
      // Notify parent component
      if (onModelChange) {
        onModelChange(model);
      }
    } catch (err) {
      console.error('Failed to switch model:', err);
      setError(language === 'ar'
        ? `فشل في التبديل إلى النموذج: ${model.name}`
        : `Failed to switch to model: ${model.name}`);
    } finally {
      setLoading(false);
    }
  };
  
  // If there's only one model or none, don't show the selector
  if (models.length <= 1 && !loading && !error) {
    return null;
  }
  
  return (
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
            sx={{ minWidth: '250px' }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column' }}>
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
              </Box>
              <Typography variant="caption" color="textSecondary">
                {model.description}
              </Typography>
            </Box>
          </MenuItem>
        ))}
      </Menu>
    </Box>
  );
};

export default ModelSelector; 