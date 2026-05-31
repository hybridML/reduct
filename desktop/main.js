const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { PythonShell } = require('python-shell');

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 900,
    minWidth: 900,
    minHeight: 700,
    title: 'Reduct',
    backgroundColor: '#0a0a0f',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'ui', 'index.html'));
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// ── IPC handlers ──────────────────────────────────────

ipcMain.handle('chat', async (event, { message, context, domain, backend, model }) => {
  const projectRoot = path.join(__dirname, '..');
  const script = path.join(projectRoot, 'proxy', 'cli.py');

  const options = {
    mode: 'json',
    pythonPath: 'python3',
    scriptPath: script,
    args: [JSON.stringify({ message, context, domain, backend, model })],
    env: { ...process.env, PYTHONPATH: projectRoot },
  };

  return new Promise((resolve, reject) => {
    PythonShell.run(script, options, (err, results) => {
      if (err) {
        reject(err.message || String(err));
        return;
      }
      if (results && results.length > 0) {
        try {
          resolve(results[results.length - 1]);
        } catch (e) {
          reject('Invalid response from backend');
        }
      } else {
        reject('No response from backend');
      }
    });
  });
});

ipcMain.handle('check-ollama', async () => {
  const http = require('http');
  return new Promise((resolve) => {
    const req = http.get('http://localhost:11434/api/tags', (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ available: true, models: parsed.models?.map(m => m.name) || [] });
        } catch {
          resolve({ available: true, models: [] });
        }
      });
    });
    req.on('error', () => resolve({ available: false, models: [] }));
    req.setTimeout(3000, () => { req.destroy(); resolve({ available: false, models: [] }); });
  });
});