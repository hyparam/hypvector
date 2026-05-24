import { describe, expect, it } from 'vitest'
import packageJson from '../package.json' with { type: 'json' }
import { hypVectorVersion } from '../src/index.js'

describe('package.json', () => {
  it('should have the correct name', () => {
    expect(packageJson.name).toBe('hypvector')
  })
  it('should have a valid version', () => {
    expect(packageJson.version).toMatch(/^\d+\.\d+\.\d+$/)
  })
  it('should have MIT license', () => {
    expect(packageJson.license).toBe('MIT')
  })
  it('should have precise dependency versions', () => {
    const { dependencies, devDependencies } = packageJson
    const allDependencies = { ...dependencies, ...devDependencies }
    Object.values(allDependencies).forEach(version => {
      expect(version).toMatch(/^\d+\.\d+\.\d+$/)
    })
  })
  it('should have minimal dependencies', () => {
    const { dependencies } = packageJson
    expect(Object.keys(dependencies).length).toBe(3)
  })
  it('hypvector version should match package.json major version', () => {
    const packageMajorVersion = parseInt(packageJson.version.split('.')[0], 10)
    expect(packageMajorVersion).toBe(hypVectorVersion)
  })
})
