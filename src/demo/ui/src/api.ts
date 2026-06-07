const API = '';

async function parseResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const message = typeof payload === 'object' && payload && 'error' in payload
      ? String((payload as { error: unknown }).error)
      : String(payload || response.statusText);
    throw new Error(message);
  }
  return payload as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  return parseResponse<T>(await fetch(API + path, { cache: 'no-store' }));
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  return parseResponse<T>(await fetch(API + path, { method: 'POST', body: form }));
}

export function boolField(value: boolean): string {
  return value ? 'true' : 'false';
}

export function formFromObject(values: Record<string, string | number | boolean | File | Blob | null | undefined>): FormData {
  const form = new FormData();
  for (const [key, value] of Object.entries(values)) {
    if (value === null || value === undefined) continue;
    if (typeof value === 'boolean') form.append(key, boolField(value));
    else if (value instanceof Blob) form.append(key, value);
    else form.append(key, String(value));
  }
  return form;
}
