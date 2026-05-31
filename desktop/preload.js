const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('reduct', {
  chat: (params) => ipcRenderer.invoke('chat', params),
  checkOllama: () => ipcRenderer.invoke('check-ollama'),
});